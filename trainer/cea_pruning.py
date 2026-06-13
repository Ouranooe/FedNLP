import logging
from typing import Dict, List, Tuple


class CHRPruningManager:
    """
    Continuous High-Response (CHR) pruning manager.

    Core rule
    ---------
    1. Each round, the server first samples a candidate set.
    2. After warmup, any sampled client with H_c >= streak_threshold is
       temporarily pruned for the current round.
    3. After aggregation, H_c is updated from the current round's LGRA
       ranking:
         * client in Top-K:     H_c = H_c + 1
         * client selected but not in Top-K: H_c = 0

    Notes
    -----
    * Pruning is temporary. Once a client is pruned for a round, its streak is
      reset so it can participate again in future rounds.
    * The manager keeps only the minimal state needed by CHR.
    """

    def __init__(
        self,
        candidate_clients: int = 8,
        warmup_rounds: int = 5,
        top_k: int = 2,
        streak_threshold: int = 2,
        max_pruned_per_round: int = 2,
    ):
        self.candidate_clients = candidate_clients
        self.warmup_rounds = warmup_rounds
        self.top_k = max(1, int(top_k))
        self.streak_threshold = max(1, int(streak_threshold))
        self.max_pruned_per_round = max(1, int(max_pruned_per_round))

        # Per-client state.
        self.H: Dict[int, int] = {}
        self.last_lgra_weight: Dict[int, float] = {}

    def select_clients(
        self,
        candidate_ids: List[int],
        round_idx: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Given *candidate_ids*, return (selected_ids, pruned_ids).

        During warmup (round_idx < warmup_rounds) all candidates are kept.
        """
        if round_idx < self.warmup_rounds:
            logging.info(
                "[CHR] round=%d warmup (<%d), keeping all %d candidates",
                round_idx,
                self.warmup_rounds,
                len(candidate_ids),
            )
            return list(candidate_ids), []

        eligible = [
            cid for cid in candidate_ids
            if self.H.get(cid, 0) >= self.streak_threshold
        ]
        if not eligible:
            return list(candidate_ids), []

        candidate_pos = {cid: idx for idx, cid in enumerate(candidate_ids)}
        eligible_sorted = sorted(
            eligible,
            key=lambda cid: (
                -self.H.get(cid, 0),
                -self.last_lgra_weight.get(cid, 0.0),
                candidate_pos[cid],
            ),
        )

        num_to_prune = min(
            self.max_pruned_per_round,
            len(eligible_sorted),
            max(0, len(candidate_ids) - 1),
        )
        if num_to_prune <= 0:
            return list(candidate_ids), []

        pruned_ids = eligible_sorted[:num_to_prune]
        pruned_set = set(pruned_ids)

        # Temporary pruning: reset the streak so the client can rejoin later.
        for cid in pruned_ids:
            self.H[cid] = 0

        selected = [cid for cid in candidate_ids if cid not in pruned_set]
        pruned = [cid for cid in candidate_ids if cid in pruned_set]

        logging.info(
            "[CHR] round=%d candidates=%s pruned=%s selected=%s",
            round_idx,
            [int(c) for c in candidate_ids],
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
        Update H_c after a training round completes.

        Args:
            selected_ids: clients that actually trained this round
            lgra_weights: {client_id: lgra_weight} from the aggregator
            round_idx: current round index
        """
        if not selected_ids:
            return

        observed = {
            cid: float(lgra_weights[cid])
            for cid in selected_ids
            if cid in lgra_weights
        }
        if not observed:
            logging.info(
                "[CHR] round=%d no LGRA weights observed, skip H update for selected=%s",
                round_idx,
                [int(c) for c in selected_ids],
            )
            return

        ranked = sorted(
            observed.keys(),
            key=lambda cid: (observed[cid], self.H.get(cid, 0)),
            reverse=True,
        )
        top_ids = set(ranked[: min(self.top_k, len(ranked))])

        for cid in selected_ids:
            weight = float(lgra_weights.get(cid, 0.0))
            self.last_lgra_weight[cid] = weight
            if cid in top_ids:
                self.H[cid] = self.H.get(cid, 0) + 1
            else:
                self.H[cid] = 0

        logging.info(
            "[CHR] round=%d top_ids=%s H=%s last_lgra=%s",
            round_idx,
            [int(c) for c in ranked[: min(self.top_k, len(ranked))]],
            {int(k): v for k, v in self.H.items()},
            {int(k): round(v, 6) for k, v in self.last_lgra_weight.items()},
        )

    def get_state_summary(self) -> Dict[str, Dict[int, float]]:
        """Return a JSON-serialisable summary for logging."""
        return {
            "H": {int(k): int(v) for k, v in self.H.items()},
            "last_lgra_weight": {
                int(k): round(v, 6) for k, v in self.last_lgra_weight.items()
            },
        }


# Backward-compatible alias so older imports still work.
CEAPruningManager = CHRPruningManager


# ------------------------------------------------------------------
# Module-level store for LGRA weights (written by aggregator, read by
# pruning manager after aggregation).
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
