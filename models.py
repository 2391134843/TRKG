"""
TRKG: Qwen2.5-based Knowledge Graph Completion via Contrastive Learning
===============================================================================
Uses shared Qwen2.5-0.5B backbone + LoRA with separate projection heads.
Gradient checkpointing enabled to reduce memory usage.
"""

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

def _patch_transformers_push_to_hub_mixin():
    """Compat patch for peft<->transformers import path differences."""
    import transformers.utils as tf_utils
    if hasattr(tf_utils, "PushToHubMixin"):
        return

    for module_name in ("transformers.utils.hub", "transformers.file_utils"):
        try:
            module = __import__(module_name, fromlist=["PushToHubMixin"])
            mixin = getattr(module, "PushToHubMixin", None)
            if mixin is not None:
                setattr(tf_utils, "PushToHubMixin", mixin)
                return
        except Exception:
            continue

def _patch_transformers_pytorch_utils():
    """Compat patch for peft expecting transformers.pytorch_utils."""
    try:
        import transformers.pytorch_utils  # noqa: F401
        return
    except Exception:
        pass

    import sys
    import types
    from transformers import modeling_utils as tf_modeling_utils

    shim = types.ModuleType("transformers.pytorch_utils")
    if hasattr(tf_modeling_utils, "Conv1D"):
        shim.Conv1D = tf_modeling_utils.Conv1D
    sys.modules["transformers.pytorch_utils"] = shim

_patch_transformers_push_to_hub_mixin()
_patch_transformers_pytorch_utils()
from peft import LoraConfig, get_peft_model, TaskType

from triplet_mask import construct_mask
from logger_config import logger

def build_model(args) -> nn.Module:
    return TRKGModel(args)

@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor

class TRKGModel(nn.Module, ABC):
    """
    Shared-backbone model using Qwen2.5 + LoRA:
      - One shared LLM encoder (saves ~50% GPU memory vs dual-encoder)
      - Separate projection heads for hr and tail
      - Gradient checkpointing for memory efficiency
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model, trust_remote_code=True)
        self.hidden_size = self.config.hidden_size

        self.log_inv_t = torch.nn.Parameter(
            torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch

        self.proj_dim = getattr(args, 'proj_dim', 256)

        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.proj_dim)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        logger.info(f'Loading shared LLM backbone: {args.pretrained_model}')
        base_model = AutoModel.from_pretrained(
            args.pretrained_model,
            trust_remote_code=True,
        )

        lora_r = getattr(args, 'lora_r', 16)
        lora_alpha = getattr(args, 'lora_alpha', 32)
        lora_dropout = getattr(args, 'lora_dropout', 0.05)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        self.encoder = get_peft_model(base_model, lora_config)

        self.encoder.gradient_checkpointing_enable()
        logger.info('Gradient checkpointing enabled')

        self.hr_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.proj_dim * 2),
            nn.GELU(),
            nn.Linear(self.proj_dim * 2, self.proj_dim),
        )
        self.tail_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.proj_dim * 2),
            nn.GELU(),
            nn.Linear(self.proj_dim * 2, self.proj_dim),
        )

        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        proj_params = sum(p.numel() for p in self.hr_proj.parameters()) + \
                      sum(p.numel() for p in self.tail_proj.parameters())
        logger.info(f'Shared encoder: {trainable:,} trainable / {total:,} total params')
        logger.info(f'Projection heads: {proj_params:,} params')
        logger.info(f'Projection dim: {self.proj_dim}')

    def _pool_output(self, last_hidden_state, attention_mask):
        """Mean pooling over non-padding tokens."""
        if self.args.pooling == 'last':
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.size(0)
            idx = torch.arange(batch_size, device=last_hidden_state.device)
            output = last_hidden_state[idx, seq_lens.long()]
        elif self.args.pooling == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-4)
            output = sum_embeddings / sum_mask
        elif self.args.pooling == 'max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
            last_hidden_state[mask_expanded == 0] = -1e4
            output = torch.max(last_hidden_state, dim=1)[0]
        else:
            raise ValueError(f'Unknown pooling mode: {self.args.pooling}')
        return output

    def _encode(self, proj_head, token_ids, mask):
        outputs = self.encoder(input_ids=token_ids, attention_mask=mask, return_dict=True)
        pooled = self._pool_output(outputs.last_hidden_state, mask)
        target_dtype = next(proj_head.parameters()).dtype
        pooled = pooled.to(target_dtype)
        projected = proj_head(pooled)
        return F.normalize(projected, dim=1)

    def forward(self, hr_token_ids, hr_mask,
                tail_token_ids, tail_mask,
                head_token_ids, head_mask,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(
                tail_token_ids=tail_token_ids, tail_mask=tail_mask)

        hr_vector = self._encode(self.hr_proj,
                                 token_ids=hr_token_ids, mask=hr_mask)
        tail_vector = self._encode(self.tail_proj,
                                   token_ids=tail_token_ids, mask=tail_mask)
        head_vector = self._encode(self.tail_proj,
                                   token_ids=head_token_ids, mask=head_mask)

        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector, tail_vector, batch_dict):
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_proj,
                                   token_ids=tail_token_ids, mask=tail_mask)
        return {'ent_vectors': ent_vectors.detach()}
