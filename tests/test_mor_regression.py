import math
import sys
import types
import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


if torch is None:
    @unittest.skip("requires torch")
    class TestMorRegression(unittest.TestCase):
        def test_torch_required(self):
            self.assertTrue(True)
else:
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.SimpleNamespace(log=lambda *args, **kwargs: None)
    try:
        from torch_main import _accumulate_update_gram, _compute_pairwise_cosine_from_gram
    except Exception:  # pragma: no cover - keep regression tests robust in minimal envs
        _accumulate_update_gram = None
        _compute_pairwise_cosine_from_gram = None

    from trainer.classification_aggregator import ClassificationAggregator
    from trainer.classification_trainer import (
        MyModelTrainer,
        clear_client_loss_deltas,
        compute_router_weights,
        reset_trained_clients,
        set_client_loss_delta,
    )

    class _DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mor_router = nn.Linear(2, 2, bias=False)
            self.backbone = nn.Linear(2, 2, bias=False)

    class TestMorRegression(unittest.TestCase):
        def setUp(self):
            reset_trained_clients()
            clear_client_loss_deltas()

        def tearDown(self):
            clear_client_loss_deltas()

        def test_mor_skip_router_first_round_upload(self):
            model = _DummyModel()
            args = SimpleNamespace(mor_skip_router_first_round=True)
            trainer = MyModelTrainer(model, args)
            trainer.id = 5

            first_upload = trainer.get_model_params()
            self.assertIn("backbone.weight", first_upload)
            self.assertNotIn("mor_router.weight", first_upload)

            second_upload = trainer.get_model_params()
            self.assertIn("mor_router.weight", second_upload)
            self.assertIn("backbone.weight", second_upload)

        def test_compute_router_weights_absolute(self):
            set_client_loss_delta(0, -0.2, first_epoch_loss=1.0)
            set_client_loss_delta(1, -0.1, first_epoch_loss=1.0)
            weights = compute_router_weights([0, 1], weight_mode="absolute", tau=1.0)

            expected_w0 = math.exp(0.2) / (math.exp(0.2) + math.exp(0.1))
            expected_w1 = 1.0 - expected_w0
            self.assertAlmostEqual(weights[0], expected_w0, places=8)
            self.assertAlmostEqual(weights[1], expected_w1, places=8)

        def test_mor_adaptive_router_aggregation_kept(self):
            set_client_loss_delta(0, -1.0, first_epoch_loss=2.0)
            set_client_loss_delta(1, 0.0, first_epoch_loss=2.0)

            args = SimpleNamespace(
                mor_adaptive_router_weight=True,
                mor_router_weight_mode="absolute",
                mor_router_weight_tau=1.0,
                mor_router_weight_tau_warmup=False,
                round_idx=0,
                model_type="mor_llama",
                enable_wandb=False,
            )
            aggregator = ClassificationAggregator(_DummyModel(), args)

            model_list = [
                (
                    1,
                    {
                        "mor_router.weight": torch.tensor([1.0]),
                        "backbone.weight": torch.tensor([10.0]),
                    },
                ),
                (
                    1,
                    {
                        "mor_router.weight": torch.tensor([3.0]),
                        "backbone.weight": torch.tensor([20.0]),
                    },
                ),
            ]

            aggregated = aggregator.aggregate(model_list, sample_num_list=[1, 1])

            expected_router_w0 = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
            expected_router_w1 = 1.0 - expected_router_w0
            expected_router_value = 1.0 * expected_router_w0 + 3.0 * expected_router_w1

            self.assertTrue(
                torch.allclose(
                    aggregated["mor_router.weight"],
                    torch.tensor([expected_router_value]),
                    atol=1e-6,
                )
            )
            self.assertTrue(
                torch.allclose(aggregated["backbone.weight"], torch.tensor([15.0]), atol=1e-6)
            )

        def test_pairwise_router_vs_backbone_cosine_stats(self):
            if _accumulate_update_gram is None or _compute_pairwise_cosine_from_gram is None:
                self.skipTest("pairwise cosine helpers unavailable")

            router_name = "mor_llama.model.layers.0.mor_router.router.weight"
            backbone_name = "mor_llama.model.layers.0.mlp.gate_proj.weight"

            global_params = {
                router_name: torch.tensor([0.0, 0.0]),
                backbone_name: torch.tensor([0.0, 0.0]),
            }
            w_locals = [
                (
                    1,
                    {
                        router_name: torch.tensor([1.0, 0.0]),
                        backbone_name: torch.tensor([1.0, 0.0]),
                    },
                ),
                (
                    1,
                    {
                        router_name: torch.tensor([0.0, 1.0]),
                        backbone_name: torch.tensor([1.0, 0.0]),
                    },
                ),
                (
                    1,
                    {
                        router_name: torch.tensor([-1.0, 0.0]),
                        backbone_name: torch.tensor([1.0, 0.0]),
                    },
                ),
            ]

            router_filter = lambda n: n.startswith("mor_llama.") and "mor_router" in n
            backbone_filter = lambda n: n.startswith("mor_llama.") and "mor_router" not in n

            router_gram, _ = _accumulate_update_gram(w_locals, global_params, router_filter)
            backbone_gram, _ = _accumulate_update_gram(w_locals, global_params, backbone_filter)

            router_stats = _compute_pairwise_cosine_from_gram(router_gram)
            backbone_stats = _compute_pairwise_cosine_from_gram(backbone_gram)

            self.assertAlmostEqual(router_stats["mean"], -1.0 / 3.0, places=6)
            self.assertAlmostEqual(backbone_stats["mean"], 1.0, places=6)
            self.assertEqual(router_stats["total_pairs"], 3)
            self.assertEqual(backbone_stats["total_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
