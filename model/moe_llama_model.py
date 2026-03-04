"""
LLaMA-based MoE Model for Sequence Classification.

This module implements a true Mixture-of-Experts architecture where each expert
is an independent LLaMA model, and a router sparsely activates top-k experts.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.utils import ModelOutput


@dataclass
class MoELlamaSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    router_logits: Optional[torch.FloatTensor] = None
    router_indices: Optional[torch.LongTensor] = None
    router_weights: Optional[torch.FloatTensor] = None
    load_balance_loss: Optional[torch.FloatTensor] = None


class MoELlamaForSequenceClassification(nn.Module):
    """
    True MoE LLaMA model for sequence classification.

    - Experts: independent LLaMA backbones (`LlamaForCausalLM`)
    - Router: linear projection + top-k sparse expert selection
    - Aggregation: weighted sum of selected expert hidden states
    """

    def __init__(
        self,
        config: LlamaConfig,
        num_labels: int = 2,
        num_experts: int = 4,
        top_k: int = 1,
        router_temperature: float = 1.0,
        load_balance_loss_coef: float = 0.0,
        weight: Optional[torch.Tensor] = None,
        pooling_type: str = "last",
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")

        self.config = config
        self.num_labels = num_labels
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_temperature = router_temperature
        self.load_balance_loss_coef = load_balance_loss_coef
        self.weight = weight
        self.pooling_type = pooling_type

        self.experts = nn.ModuleList([LlamaForCausalLM(config) for _ in range(num_experts)])
        self.router = nn.Linear(config.hidden_size, num_experts, bias=False)

        self.dropout = nn.Dropout(
            config.hidden_dropout_prob if hasattr(config, "hidden_dropout_prob") else 0.1
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        nn.init.normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def load_pretrained_experts(self, model_name_or_path: str, **kwargs):
        pretrained = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        pretrained_state_dict = pretrained.state_dict()
        for expert in self.experts:
            expert.load_state_dict(pretrained_state_dict, strict=False)

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pooling_type == "last":
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                return hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    sequence_lengths,
                ]
            return hidden_states[:, -1, :]

        if self.pooling_type == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_hidden / sum_mask
            return hidden_states.mean(dim=1)

        return hidden_states[:, -1, :]

    def _build_router_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if inputs_embeds is not None:
            token_embeds = inputs_embeds
        elif input_ids is not None:
            token_embeds = self.experts[0].model.embed_tokens(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        if attention_mask is None:
            return token_embeds.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeds * mask).sum(dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom

    def _compute_load_balance_loss(
        self,
        router_probs: torch.FloatTensor,
        topk_indices: torch.LongTensor,
    ) -> torch.FloatTensor:
        batch_size, num_experts = router_probs.shape
        importance = router_probs.mean(dim=0)

        selected = torch.zeros(batch_size, num_experts, device=router_probs.device)
        selected.scatter_(1, topk_indices, 1.0)
        load = selected.mean(dim=0)

        return num_experts * torch.sum(importance * load)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Tuple:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds must be provided")

        router_inputs = self._build_router_inputs(input_ids, inputs_embeds, attention_mask)
        router_logits = self.router(router_inputs) / max(self.router_temperature, 1e-6)
        router_probs = torch.softmax(router_logits, dim=-1)

        topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)

        batch_size = router_logits.size(0)

        if input_ids is not None:
            seq_len = input_ids.size(1)
        else:
            seq_len = inputs_embeds.size(1)

        hidden_size = self.config.hidden_size
        mixed_hidden_states = torch.zeros(
            batch_size,
            seq_len,
            hidden_size,
            device=router_logits.device,
            dtype=self.experts[0].model.embed_tokens.weight.dtype,
        )

        for expert_id in range(self.num_experts):
            selected_mask = (topk_indices == expert_id).any(dim=-1)
            if not torch.any(selected_mask):
                continue

            selected_rows = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)

            expert_input_ids = input_ids[selected_rows] if input_ids is not None else None
            expert_inputs_embeds = inputs_embeds[selected_rows] if inputs_embeds is not None else None
            expert_attention_mask = attention_mask[selected_rows] if attention_mask is not None else None
            expert_position_ids = position_ids[selected_rows] if position_ids is not None else None

            expert_outputs = self.experts[expert_id].model(
                input_ids=expert_input_ids,
                attention_mask=expert_attention_mask,
                position_ids=expert_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=expert_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs,
            )
            expert_hidden = expert_outputs.last_hidden_state

            gate = (topk_weights[selected_rows] * (topk_indices[selected_rows] == expert_id).float()).sum(dim=-1)
            expert_hidden = expert_hidden * gate.view(-1, 1, 1)

            mixed_hidden_states[selected_rows] += expert_hidden

        pooled_output = self._pool_hidden_states(mixed_hidden_states, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        load_balance_loss = self._compute_load_balance_loss(router_probs, topk_indices)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.load_balance_loss_coef > 0:
                loss = loss + self.load_balance_loss_coef * load_balance_loss

        output_tuple = (logits,)
        if loss is not None:
            output_tuple = (loss,) + output_tuple

        return output_tuple
