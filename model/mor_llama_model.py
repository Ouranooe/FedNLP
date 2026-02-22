"""
MoR (Mixture of Recursions) LLaMA Model for Sequence Classification

This module integrates the MoR model into the FedNLP framework for text classification,
following the same interface as BertForSequenceClassification.
"""

import os
import sys

# Add MoR package path to sys.path for proper imports
# This enables MoR internal imports like `from model.xxx` to work correctly
_mor_package_path = os.path.join(os.path.dirname(__file__), 'mor')
if _mor_package_path not in sys.path:
    sys.path.insert(0, _mor_package_path)

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

from transformers import LlamaConfig
from transformers.utils import ModelOutput

# Import MoR model components (these use internal `from model.xxx` imports)
from model.mor.model_mor.mor_model.modeling_llama import MoRLlamaModel, MoRLlamaForCausalLM
from model.mor.model_mor.sharing_strategy.llama import average_initialize


@dataclass
class MoRSequenceClassifierOutput(ModelOutput):
    """
    Output type for MoR sequence classification models.
    
    Args:
        loss: Classification loss (includes MoR auxiliary losses)
        logits: Classification logits
        hidden_states: Hidden states from the model
        attentions: Attention weights
        mor_loss: Combined MoR auxiliary loss (sampling + balancing + router_z)
        sampling_loss: Sampling loss from expert router
        balancing_loss: Balancing loss from token router
        router_z_loss: Router z-loss for regularization
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    mor_loss: Optional[torch.FloatTensor] = None
    sampling_loss: Optional[torch.FloatTensor] = None
    balancing_loss: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None


class MoRLlamaForSequenceClassification(nn.Module):
    """
    MoR LLaMA Model for Sequence Classification.
    
    This model uses the MoR (Mixture of Recursions) architecture with LLaMA backbone
    for text classification tasks. It pools the last hidden state and applies a 
    classification head.
    
    The model forward returns a tuple compatible with FedNLP trainer:
    (loss, logits, hidden_states, attentions)
    
    MoR auxiliary losses (sampling_loss, balancing_loss, router_z_loss) are added
    to the classification loss to stabilize MoR behavior.
    """
    
    def __init__(
        self, 
        config: LlamaConfig, 
        num_labels: int = 2,
        mor_config: Optional[DictConfig] = None,
        weight: Optional[torch.Tensor] = None,
        mor_loss_coeff: float = 0.01,
    ):
        """
        Initialize MoR LLaMA for Sequence Classification.
        
        Args:
            config: LlamaConfig for the base model
            num_labels: Number of classification labels
            mor_config: MoR configuration (OmegaConf DictConfig)
            weight: Optional class weights for CrossEntropyLoss
            mor_loss_coeff: Coefficient for MoR auxiliary losses in total loss
        """
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        self.mor_config = mor_config
        self.weight = weight
        self.mor_loss_coeff = mor_loss_coeff
        
        # Initialize MoR LLaMA backbone
        self.mor_llama = MoRLlamaForCausalLM(config)
        
        # Classification head
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Pooling type: 'last' uses last token, 'mean' uses mean pooling
        self.pooling_type = 'last'
        
        # Initialize classification head weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize the classifier weights."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    @classmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str,
        num_labels: int = 2,
        mor_config: Optional[DictConfig] = None,
        **kwargs
    ):
        """
        Load a pretrained MoR LLaMA model and add classification head.
        
        Args:
            model_name_or_path: Path to pretrained model or model identifier
            num_labels: Number of classification labels
            mor_config: MoR configuration
            **kwargs: Additional arguments passed to model loading
        
        Returns:
            MoRLlamaForSequenceClassification instance
        """
        config = LlamaConfig.from_pretrained(model_name_or_path)
        config.num_labels = num_labels
        
        model = cls(config, num_labels=num_labels, mor_config=mor_config, **kwargs)
        
        # Load pretrained weights for the backbone
        pretrained_model = MoRLlamaForCausalLM.from_pretrained(model_name_or_path)
        model.mor_llama.load_state_dict(pretrained_model.state_dict(), strict=False)
        
        return model
    
    def setup_mor(self, cfg: DictConfig):
        """
        Setup MoR (Mixture of Recursions) components.
        
        This transforms the model layers according to MoR configuration,
        including expert routing and token routing strategies.
        
        Args:
            cfg: OmegaConf DictConfig containing MoR configuration
        """
        self.mor_config = cfg
        
        # Apply sharing strategy initialization if enabled
        if hasattr(cfg, 'recursive') and cfg.recursive.enable:
            self.mor_llama, _ = average_initialize(cfg, self.mor_llama)
        
        # Transform layers to MoR expert or token routing
        if hasattr(cfg, 'mor') and cfg.mor.enable:
            if cfg.mor.type == 'expert':
                self.mor_llama.transform_layer_to_mor_expert(cfg)
            elif cfg.mor.type == 'token':
                self.mor_llama.transform_layer_to_mor_token(cfg)
        
        # Set KV sharing config if enabled
        if hasattr(cfg, 'kv_sharing') and cfg.kv_sharing.enable:
            self.mor_llama.set_kv_sharing_config(cfg)
    
    def _pool_hidden_states(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool the hidden states for classification.
        
        Args:
            hidden_states: Hidden states of shape (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_length)
        
        Returns:
            Pooled representation of shape (batch_size, hidden_size)
        """
        if self.pooling_type == 'last':
            # Use the last non-padding token
            if attention_mask is not None:
                # Find the last non-padding position for each sequence
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                pooled = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    sequence_lengths
                ]
            else:
                pooled = hidden_states[:, -1, :]
        elif self.pooling_type == 'mean':
            # Mean pooling over non-padding tokens
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)
        else:
            # Default: use last token
            pooled = hidden_states[:, -1, :]
        
        return pooled
    
    def _compute_mor_loss(self, outputs) -> torch.Tensor:
        """
        Compute the combined MoR auxiliary loss.
        
        Args:
            outputs: Model outputs containing sampling_loss, balancing_loss, router_z_loss
        
        Returns:
            Combined MoR auxiliary loss
        """
        mor_loss = torch.tensor(0.0, device=outputs.last_hidden_state.device)
        
        if hasattr(outputs, 'sampling_loss') and outputs.sampling_loss is not None:
            mor_loss = mor_loss + outputs.sampling_loss
        
        if hasattr(outputs, 'balancing_loss') and outputs.balancing_loss is not None:
            mor_loss = mor_loss + outputs.balancing_loss
        
        if hasattr(outputs, 'router_z_loss') and outputs.router_z_loss is not None:
            mor_loss = mor_loss + outputs.router_z_loss
        
        return mor_loss
    
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
        """
        Forward pass for sequence classification.
        
        Returns a tuple compatible with FedNLP trainer:
        (loss, logits, hidden_states, attentions)
        
        If labels are provided, loss includes:
        - Classification loss (CrossEntropy or MSE)
        - MoR auxiliary losses (sampling_loss + balancing_loss + router_z_loss)
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for caching
            inputs_embeds: Input embeddings (alternative to input_ids)
            labels: Classification labels
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict (ignored, always returns tuple)
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (loss, logits, hidden_states, attentions)
        """
        # Get outputs from MoR LLaMA model (access the inner model, not the causal LM)
        outputs = self.mor_llama.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Pool the hidden states
        pooled_output = self._pool_hidden_states(hidden_states, attention_mask)
        pooled_output = self.dropout(pooled_output)
        
        # Classification logits
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels are provided
        loss = None
        mor_loss = None
        
        if labels is not None:
            # Classification loss
            if self.num_labels == 1:
                # Regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # Classification
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # Add MoR auxiliary losses
            mor_loss = self._compute_mor_loss(outputs)
            if mor_loss.item() > 0:
                loss = loss + self.mor_loss_coeff * mor_loss
        
        # Build output tuple compatible with FedNLP trainer: (loss, logits, ...)
        output_tuple = (logits,)
        
        # Add hidden states if available
        if output_hidden_states and outputs.hidden_states is not None:
            output_tuple = output_tuple + (outputs.hidden_states,)
        
        # Add attentions if available
        if output_attentions and outputs.attentions is not None:
            output_tuple = output_tuple + (outputs.attentions,)
        
        # Prepend loss if labels were provided
        if loss is not None:
            output_tuple = (loss,) + output_tuple
        
        return output_tuple


def create_mor_config(args) -> DictConfig:
    """
    Create MoR configuration from FedML args.
    
    This function constructs an OmegaConf DictConfig compatible with MoR model
    from the FedML training arguments.
    
    Args:
        args: FedML training arguments
    
    Returns:
        OmegaConf DictConfig for MoR configuration
    """
    # Default MoR configuration
    mor_cfg = {
        'recursive': {
            'enable': getattr(args, 'recursive_enable', True),
            'base_depth': getattr(args, 'recursive_base_depth', None),
            'num_recursion': getattr(args, 'recursive_num_recursion', 3),
            'sharing': getattr(args, 'recursive_sharing', 'middle_cycle'),
            'ln_share': getattr(args, 'recursive_ln_share', True),
            'initialization': getattr(args, 'recursive_initialization', 'random'),
        },
        'kv_sharing': {
            'enable': getattr(args, 'kv_sharing_enable', False),
            'base_depth': getattr(args, 'kv_sharing_base_depth', None),
            'num_recursion': getattr(args, 'kv_sharing_num_recursion', None),
            'sharing': getattr(args, 'kv_sharing_sharing', None),
        },
        'mor': {
            'enable': getattr(args, 'mor_enable', True),
            'type': getattr(args, 'mor_type', 'expert'),
            'capacity': getattr(args, 'mor_capacity', '1.0,1.0,1.0'),
            'rand_router': getattr(args, 'mor_rand_router', False),
            'router_type': getattr(args, 'mor_router_type', 'linear'),
            'z_loss': getattr(args, 'mor_z_loss', False),
            'z_coeff': getattr(args, 'mor_z_coeff', 0.001),
            'temp': getattr(args, 'mor_temp', 1.0),
            'expert': {
                'cap_warmup_step': getattr(args, 'mor_expert_cap_warmup_step', 0),
                'router_func': getattr(args, 'mor_expert_router_func', 'sigmoid'),
                'alpha': getattr(args, 'mor_expert_alpha', 0.1),
                'sampling': getattr(args, 'mor_expert_sampling', 'aux_loss'),
                'include_first': getattr(args, 'mor_expert_include_first', True),
                'coeff': getattr(args, 'mor_expert_coeff', 0.001),
                'gating': getattr(args, 'mor_expert_gating', 'weighted'),
            },
            'token': {
                'router_func': getattr(args, 'mor_token_router_func', 'sigmoid'),
                'alpha': getattr(args, 'mor_token_alpha', 0.1),
                'balancing': getattr(args, 'mor_token_balancing', 'loss'),
                'coeff': getattr(args, 'mor_token_coeff', 0.001),
                'u': getattr(args, 'mor_token_u', 0.001),
                'gating': getattr(args, 'mor_token_gating', 'weighted'),
                'bal_warmup_step': getattr(args, 'mor_token_bal_warmup_step', 0),
            },
        },
        # Training related configs needed by MoR
        'num_warmup_steps': getattr(args, 'num_warmup_steps', 100),
        'gradient_accumulation_steps': getattr(args, 'gradient_accumulation_steps', 1),
        # Precision setting from FedML
        'precision': 'fp16' if getattr(args, 'fp16', False) else 'fp32',
    }
    
    return OmegaConf.create(mor_cfg)
