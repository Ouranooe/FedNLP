import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

try:
    from transformers import LlamaConfig
except ModuleNotFoundError:
    LlamaConfig = None

if torch is None:
    @unittest.skip("requires torch")
    class TestMoERegression(unittest.TestCase):
        def test_torch_required(self):
            self.assertTrue(True)
else:
    from model.moe_llama_model import MoELlamaForSequenceClassification
    from trainer.classification_trainer import MyModelTrainer

    class TestMoERegression(unittest.TestCase):
        def test_top1_router_still_gets_gradient(self):
            if LlamaConfig is None:
                self.skipTest("requires transformers")

            torch.manual_seed(7)
            config = LlamaConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            model = MoELlamaForSequenceClassification(
                config=config,
                num_labels=3,
                num_experts=3,
                top_k=1,
                router_temperature=1.0,
                load_balance_loss_coef=0.001,
            )

            input_ids = torch.randint(3, 120, (4, 8), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = torch.randint(0, 3, (4,), dtype=torch.long)

            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()

            router_grad = model.router.weight.grad
            self.assertIsNotNone(router_grad)
            self.assertTrue(torch.isfinite(router_grad).all().item())
            self.assertGreater(float(router_grad.abs().sum().item()), 0.0)

        def test_trainer_uses_model_loss_for_moe(self):
            class DummyMoEModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = nn.Parameter(torch.tensor(1.0))

                def forward(self, input_ids, attention_mask=None, labels=None):
                    logits = torch.stack(
                        [self.scale.repeat(input_ids.size(0)), (-self.scale).repeat(input_ids.size(0))],
                        dim=1,
                    )
                    internal_loss = self.scale * 3.0
                    return internal_loss, logits

            class DummyBaseModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = nn.Parameter(torch.tensor(1.0))

                def forward(self, input_ids, attention_mask=None):
                    return (torch.stack(
                        [self.scale.repeat(input_ids.size(0)), (-self.scale).repeat(input_ids.size(0))],
                        dim=1,
                    ),)

            x = torch.ones((2, 5), dtype=torch.long)
            attention_mask = torch.ones_like(x)
            labels = torch.tensor([0, 1], dtype=torch.long)
            criterion = torch.nn.CrossEntropyLoss()

            moe_args = SimpleNamespace(model_type="moe_llama", model_class="transformer")
            moe_model = DummyMoEModel()
            moe_trainer = MyModelTrainer(moe_model, moe_args)
            moe_loss, moe_logits = moe_trainer._forward_and_compute_loss(
                model=moe_model,
                x=x,
                attention_mask=attention_mask,
                labels=labels,
                criterion=criterion,
                args=moe_args,
            )
            external_moe_loss = criterion(moe_logits, labels)
            self.assertAlmostEqual(float(moe_loss.item()), 3.0, places=6)
            self.assertNotAlmostEqual(float(moe_loss.item()), float(external_moe_loss.item()), places=6)

            base_args = SimpleNamespace(model_type="llama", model_class="transformer")
            base_model = DummyBaseModel()
            base_trainer = MyModelTrainer(base_model, base_args)
            base_loss, base_logits = base_trainer._forward_and_compute_loss(
                model=base_model,
                x=x,
                attention_mask=attention_mask,
                labels=labels,
                criterion=criterion,
                args=base_args,
            )
            expected_base_loss = criterion(base_logits, labels)
            self.assertAlmostEqual(float(base_loss.item()), float(expected_base_loss.item()), places=6)


if __name__ == "__main__":
    unittest.main()
