import logging
from typing import Dict, List, Optional, Tuple


class CEAPruningManager:
    """
    Communication-Efficient Adaptive Pruning (CEA-Pruning) manager.

    Reduces the number of fully-communicating clients per round by pruning
    low-utility candidates based on historical LGRA weights, using an
    Oort-style utility-fairness score.

    Maintained per-client state
    --------------------------
    A_c : float          EMA of historical LGRA weight
    N_c : int            total number of rounds this client has participated
    last_selected_round : int   last round in which this client was selected

    Pruning score (Oort-style)
    --------------------------
    U_c     = A_c
    F_c     = max_A - A_c
    score_c = (1 - f) * U_c + f * F_c

    Protection rules
    ----------------
    * N_c == 0 → cannot be pruned  (never participated)
    * current_round - last_selected_round >= max_stale_rounds → cannot be pruned
    """

    def __init__(
        self,
        candidate_clients: int = 8,
        keep_clients: int = 6,
        warmup_rounds: int = 5,
        ema_beta: float = 0.9,
        fairness_f: float = 0.3,
        max_stale_rounds: int = 6,
    ):
        self.candidate_clients = candidate_clients
        self.keep_clients = keep_clients
        self.warmup_rounds = warmup_rounds
        self.ema_beta = ema_beta
        self.fairness_f = fairness_f
        self.max_stale_rounds = max_stale_rounds

        # per-client state: client_id -> value
        self.A: Dict[int, float] = {}          # EMA of LGRA weight
        self.N: Dict[int, int] = {}            # participation count
        self.last_selected: Dict[int, int] = {}  # last selected round

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_clients(
        self,
        candidate_ids: List[int],
        round_idx: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Given *candidate_ids* (size == candidate_clients), return
        (selected_ids, pruned_ids).

        During warmup (round_idx < warmup_rounds) all candidates are kept.
        """
        if round_idx < self.warmup_rounds:
            logging.info(
                "[CEA] round=%d warmup (<%d), keeping all %d candidates",
                round_idx, self.warmup_rounds, len(candidate_ids),
            )
            return list(candidate_ids), []

        if len(candidate_ids) <= self.keep_clients:
            return list(candidate_ids), []

        # --- identify protected clients ---
        protected = set()
        for cid in candidate_ids:
            if self.N.get(cid, 0) == 0:
                protected.add(cid)
            elif round_idx - self.last_selected.get(cid, -self.max_stale_rounds) >= self.max_stale_rounds:
                protected.add(cid)

        # --- compute scores for prunable candidates ---
        prunable = [cid for cid in candidate_ids if cid not in protected]
        a_values = [self.A.get(cid, 0.0) for cid in prunable]
        max_a = max(a_values) if a_values else 0.0

        scores: Dict[int, float] = {}
        for cid in prunable:
            a_c = self.A.get(cid, 0.0)
            u_c = a_c
            f_c = max_a - a_c
            scores[cid] = (1 - self.fairness_f) * u_c + self.fairness_f * f_c

        # --- decide how many to prune ---
        num_to_prune = len(candidate_ids) - self.keep_clients
        # cannot prune more than the prunable set
        num_to_prune = min(num_to_prune, len(prunable))
        # if all are protected, nothing to prune
        if num_to_prune <= 0:
            logging.info(
                "[CEA] round=%d all candidates protected or keep>=candidates, no pruning",
                round_idx,
            )
            return list(candidate_ids), []

        # sort prunable by score ascending (lowest score gets pruned)
        prunable_sorted = sorted(prunable, key=lambda cid: scores[cid])
        pruned_ids = prunable_sorted[:num_to_prune]
        pruned_set = set(pruned_ids)

        selected = [cid for cid in candidate_ids if cid not in pruned_set]
        pruned = [cid for cid in candidate_ids if cid in pruned_set]

        logging.info(
            "[CEA] round=%d candidates=%s protected=%s pruned=%s selected=%s",
            round_idx,
            [int(c) for c in candidate_ids],
            [int(c) for c in sorted(protected)],
            [int(c) for c in pruned],
            [int(c) for c in selected],
        )
        return selected, pruned

    def update_after_round(
        self,
        selected_ids: List[int],
        lgra_weights: Dict[int, float],
        round_idx: int,
    ) -> None:
        """
        Update EMA, participation count, and last_selected_round after a
        training round completes.

        Args:
            selected_ids: clients that actually trained this round
            lgra_weights: {client_id: lgra_weight} from the aggregator
            round_idx: current round index
        """
        for cid in selected_ids:
            # update participation count
            self.N[cid] = self.N.get(cid, 0) + 1
            # update last selected round
            self.last_selected[cid] = round_idx
            # update EMA of LGRA weight
            w = lgra_weights.get(cid, None)
            if w is not None:
                old_a = self.A.get(cid, w)  # init with first observation
                self.A[cid] = self.ema_beta * old_a + (1 - self.ema_beta) * w
            # else: no lgra weight for this client (e.g. warmup), keep old A

        logging.info(
            "[CEA] round=%d EMA update: A=%s N=%s",
            round_idx,
            {int(k): round(v, 6) for k, v in self.A.items()},
            {int(k): v for k, v in self.N.items()},
        )

    def get_state_summary(self) -> Dict:
        """Return a JSON-serialisable summary for logging."""
        return {
            "A": {int(k): round(v, 6) for k, v in self.A.items()},
            "N": {int(k): v for k, v in self.N.items()},
            "last_selected": {int(k): v for k, v in self.last_selected.items()},
        }


# ------------------------------------------------------------------
# Module-level store for LGRA weights (written by aggregator, read by
# CEA manager after aggregation).
# ------------------------------------------------------------------
_last_lgra_weights: Dict[int, float] = {}


def store_lgra_weights(weights: Dict[int, float]) -> None:
    global _last_lgra_weights
    _last_lgra_weights = dict(weights)


def pop_lgra_weights() -> Dict[int, float]:
    global _last_lgra_weights
    w = _last_lgra_weights
    _last_lgra_weights = {}
    return w
