"""
Base LLaMA Model for Sequence Classification

This module integrates the standard LLaMA model (without MoR) into the FedNLP framework 
for text classification, serving as a baseline model for comparison with FedMoR.
Following the same interface as BertForSequenceClassification and MoRLlamaForSequenceClassification.
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.utils import ModelOutput


@dataclass
class LlamaSequenceClassifierOutput(ModelOutput):
    """
    Output type for LLaMA sequence classification models.
    
    Args:
        loss: Classification loss
        logits: Classification logits
        hidden_states: Hidden states from the model
        attentions: Attention weights
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class LlamaForSequenceClassification(nn.Module):
    """
    LLaMA Model for Sequence Classification.
    
    This is the base LLaMA model (without MoR modifications) for text classification tasks.
    It pools the last hidden state and applies a classification head.
    
    The model forward returns a tuple compatible with FedNLP trainer:
    (loss, logits, hidden_states, attentions)
    
    This serves as a baseline model for comparison with FedMoR.
    """
    
    def __init__(
        self,
        config: LlamaConfig,
        num_labels: int = 2,
        weight: Optional[torch.Tensor] = None,
        pooling_type: str = 'last',
    ):
        """
        Initialize LLaMA for Sequence Classification.
        
        Args:
            config: LlamaConfig for the base model
            num_labels: Number of classification labels
            weight: Optional class weights for CrossEntropyLoss
            pooling_type: Pooling strategy ('last' or 'mean')
        """
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        self.weight = weight
        self.pooling_type = pooling_type
        
        # Initialize LLaMA backbone (using the CausalLM for pretrained weight compatibility)
        self.llama = LlamaForCausalLM(config)
        
        # Classification head
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Store torch dtype for consistency (default fp32, let AMP handle mixed precision)
        self.torch_dtype = torch.float32
        
        # Initialize classification head weights
        self._init_classifier_weights()
    
    def half(self):
        """Convert model to half precision (fp16).
        Note: With AMP, manual precision conversion is not recommended.
        """
        super().half()
        self.torch_dtype = torch.float16
        return self
    
    def float(self):
        """Convert model to float precision (fp32).
        Note: With AMP, manual precision conversion is not recommended.
        """
        super().float()
        self.torch_dtype = torch.float32
        return self
    
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
        **kwargs
    ):
        """
        Load a pretrained LLaMA model and add classification head.
        
        Args:
            model_name_or_path: Path to pretrained model or model identifier
            num_labels: Number of classification labels
            **kwargs: Additional arguments passed to model loading
        
        Returns:
            LlamaForSequenceClassification instance
        """
        config = LlamaConfig.from_pretrained(model_name_or_path)
        config.num_labels = num_labels
        
        # Extract weight argument if present
        weight = kwargs.pop('weight', None)
        pooling_type = kwargs.pop('pooling_type', 'last')
        
        model = cls(config, num_labels=num_labels, weight=weight, pooling_type=pooling_type)
        
        # Load pretrained weights for the backbone
        pretrained_model = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        model.llama.load_state_dict(pretrained_model.state_dict(), strict=False)
        
        return model
    
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
            # Use the last non-padding token (for decoder-only models like LLaMA)
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
        # Get outputs from LLaMA model (access the inner model, not the causal LM)
        outputs = self.llama.model(
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
    
    def get_input_embeddings(self):
        """Get input embeddings from the model."""
        return self.llama.model.embed_tokens
    
    def set_input_embeddings(self, value):
        """Set input embeddings for the model."""
        self.llama.model.embed_tokens = value
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """
        Resize the token embeddings.
        
        Args:
            new_num_tokens: New number of tokens
            
        Returns:
            The resized embedding layer
        """
        return self.llama.resize_token_embeddings(new_num_tokens)
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the model."""
        self.llama.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        self.llama.gradient_checkpointing_disable()
    
    def get_num_params(self, only_trainable: bool = False) -> int:
        """
        Get the number of parameters in the model.
        
        Args:
            only_trainable: If True, only count trainable parameters
            
        Returns:
            Number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def freeze_backbone(self):
        """Freeze the LLaMA backbone, only train the classifier head."""
        for param in self.llama.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze the LLaMA backbone for full fine-tuning."""
        for param in self.llama.parameters():
            param.requires_grad = True
