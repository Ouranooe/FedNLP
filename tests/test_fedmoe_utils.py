import unittest
from types import SimpleNamespace

from trainer.fedmoe_utils import (
    aggregate_fedmoe_params,
    filter_state_dict_for_experts,
    get_client_expert_assignment,
    get_expert_id_from_param_name,
    get_fedmoe_config,
)


class TestFedMoEUtils(unittest.TestCase):
    def test_assignment_is_deterministic(self):
        a1 = get_client_expert_assignment(
            client_id=7, num_experts=4, experts_per_client=2, assignment_seed=123
        )
        a2 = get_client_expert_assignment(
            client_id=7, num_experts=4, experts_per_client=2, assignment_seed=123
        )
        self.assertEqual(a1, a2)

    def test_param_group_parsing(self):
        self.assertEqual(get_expert_id_from_param_name("experts.0.model.layers.0.mlp.gate_proj.weight"), 0)
        self.assertEqual(get_expert_id_from_param_name("experts.12.foo"), 12)
        self.assertIsNone(get_expert_id_from_param_name("router.weight"))
        self.assertIsNone(get_expert_id_from_param_name("classifier.bias"))

    def test_filter_state_dict_for_client_assignment(self):
        state = {
            "router.weight": 1.0,
            "classifier.weight": 2.0,
            "experts.0.x": 10.0,
            "experts.1.x": 20.0,
        }
        filtered = filter_state_dict_for_experts(state, assigned_experts=[1])
        self.assertIn("router.weight", filtered)
        self.assertIn("classifier.weight", filtered)
        self.assertIn("experts.1.x", filtered)
        self.assertNotIn("experts.0.x", filtered)

    def test_aggregate_shared_and_experts(self):
        w_locals = [
            (
                2,
                {
                    "router.weight": 2.0,
                    "experts.0.w": 10.0,
                },
            ),
            (
                1,
                {
                    "router.weight": 4.0,
                    "experts.1.w": 30.0,
                },
            ),
        ]
        prev_global = {
            "router.weight": 0.0,
            "experts.0.w": 100.0,
            "experts.1.w": 200.0,
        }
        aggregated, stats = aggregate_fedmoe_params(
            w_locals=w_locals,
            previous_global_params=prev_global,
            num_experts=2,
            keep_expert_if_no_contributor=True,
        )
        self.assertAlmostEqual(aggregated["router.weight"], 8.0 / 3.0, places=8)
        self.assertAlmostEqual(aggregated["experts.0.w"], 10.0, places=8)
        self.assertAlmostEqual(aggregated["experts.1.w"], 30.0, places=8)
        self.assertEqual(stats["expert_contributors"][0], 1)
        self.assertEqual(stats["expert_contributors"][1], 1)

    def test_keep_previous_expert_when_no_contributor(self):
        w_locals = [
            (2, {"router.weight": 2.0, "experts.0.w": 10.0}),
            (1, {"router.weight": 4.0, "experts.0.w": 14.0}),
        ]
        prev_global = {
            "router.weight": 0.0,
            "experts.0.w": 100.0,
            "experts.1.w": 200.0,
        }
        aggregated, stats = aggregate_fedmoe_params(
            w_locals=w_locals,
            previous_global_params=prev_global,
            num_experts=2,
            keep_expert_if_no_contributor=True,
        )
        self.assertAlmostEqual(aggregated["experts.1.w"], 200.0, places=8)
        self.assertEqual(stats["expert_contributors"][1], 0)

    def test_config_parse_from_nested_moe_args(self):
        args = SimpleNamespace(
            model_type="moe_llama",
            moe_args={
                "fedmoe_enable": True,
                "moe_num_experts": 8,
                "fedmoe_experts_per_client": 3,
                "fedmoe_assignment_seed": 9,
                "fedmoe_keep_expert_if_no_contributor": False,
            },
        )
        cfg = get_fedmoe_config(args)
        self.assertTrue(cfg["fedmoe_enable"])
        self.assertEqual(cfg["num_experts"], 8)
        self.assertEqual(cfg["experts_per_client"], 3)
        self.assertEqual(cfg["assignment_seed"], 9)
        self.assertFalse(cfg["keep_expert_if_no_contributor"])


if __name__ == "__main__":
    unittest.main()
