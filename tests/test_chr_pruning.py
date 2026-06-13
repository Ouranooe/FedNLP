import unittest

from trainer.cea_pruning import CHRPruningManager, pop_lgra_weights, store_lgra_weights


class TestCHRPruning(unittest.TestCase):
    def test_warmup_keeps_all_candidates(self):
        manager = CHRPruningManager(
            candidate_clients=8,
            warmup_rounds=2,
            top_k=2,
            streak_threshold=2,
            max_pruned_per_round=2,
        )

        selected, pruned = manager.select_clients([0, 1, 2, 3], round_idx=1)

        self.assertEqual(selected, [0, 1, 2, 3])
        self.assertEqual(pruned, [])

    def test_selected_non_top_clients_reset_streak(self):
        manager = CHRPruningManager(
            candidate_clients=8,
            warmup_rounds=0,
            top_k=2,
            streak_threshold=2,
            max_pruned_per_round=2,
        )

        manager.update_after_round([0, 1, 2], {0: 0.6, 1: 0.3, 2: 0.1}, round_idx=0)
        self.assertEqual(manager.H[0], 1)
        self.assertEqual(manager.H[1], 1)
        self.assertEqual(manager.H[2], 0)

        manager.update_after_round([0, 1, 2], {2: 0.8, 1: 0.2, 0: 0.1}, round_idx=1)
        self.assertEqual(manager.H[0], 0)
        self.assertEqual(manager.H[1], 2)
        self.assertEqual(manager.H[2], 1)

    def test_streak_based_pruning_is_temporary(self):
        manager = CHRPruningManager(
            candidate_clients=8,
            warmup_rounds=0,
            top_k=2,
            streak_threshold=2,
            max_pruned_per_round=2,
        )

        manager.update_after_round([0, 1, 2, 3], {0: 0.5, 1: 0.4, 2: 0.2, 3: 0.1}, round_idx=0)
        manager.update_after_round([0, 1, 2, 3], {0: 0.6, 1: 0.45, 2: 0.2, 3: 0.1}, round_idx=1)

        selected, pruned = manager.select_clients([0, 1, 2, 3], round_idx=2)
        self.assertEqual(pruned, [0, 1])
        self.assertEqual(selected, [2, 3])
        self.assertEqual(manager.H[0], 0)
        self.assertEqual(manager.H[1], 0)

        selected_again, pruned_again = manager.select_clients([0, 1, 2, 3], round_idx=3)
        self.assertEqual(selected_again, [0, 1, 2, 3])
        self.assertEqual(pruned_again, [])

    def test_lgra_weight_store_round_trip(self):
        store_lgra_weights({1: 0.7, 3: 0.3})
        self.assertEqual(pop_lgra_weights(), {1: 0.7, 3: 0.3})
        self.assertEqual(pop_lgra_weights(), {})


if __name__ == "__main__":
    unittest.main()
