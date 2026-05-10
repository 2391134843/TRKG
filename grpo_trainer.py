"""
GRPO (Group Relative Policy Optimization) Trainer for Chat-based KGC Reranking.

Memory optimizations:
  - bfloat16 for both policy and reference models
  - gradient checkpointing on policy
  - selective logit computation: only project to option tokens (~10) instead of full vocab (151K)

Algorithm:
  For each prompt x with K candidate options:
    1. Forward pass to get hidden states at the last position
    2. Project hidden states ONLY onto option token embeddings (A, B, C, ...)
    3. Sample G responses from the categorical distribution
    4. Compute rewards: r=1 if correct, r=0 otherwise
    5. Compute group-normalized advantages: A_i = (r_i - mean(r)) / (std(r) + eps)
    6. GRPO loss = -mean(A_i * log_pi(y_i|x)) + beta * KL(pi_theta || pi_ref)
    7. Backward + update
"""
import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

import deepspeed

from logger_config import logger
from utils import AverageMeter, ProgressMeter

class GRPOTrainer:

    def __init__(self, args):
        self.args = args
        self.use_deepspeed = bool(getattr(args, 'deepspeed_config', ''))
        self.local_rank = getattr(args, 'local_rank', -1)
        self.is_main_process = (self.local_rank <= 0)

        if self.use_deepspeed and self.local_rank >= 0:
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.is_main_process:
            logger.info(f'Loading policy model: {args.chat_model}')
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.chat_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.policy = AutoModelForCausalLM.from_pretrained(
            args.chat_model, trust_remote_code=True,
            torch_dtype=torch.bfloat16)

        lora_config = LoraConfig(
            r=args.grpo_lora_r,
            lora_alpha=args.grpo_lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.policy = get_peft_model(self.policy, lora_config)
        self.policy.enable_input_require_grads()
        self.policy.gradient_checkpointing_enable()
        self.policy.print_trainable_parameters()

        ref_quantize = getattr(args, 'ref_model_quantize', '')
        if self.is_main_process:
            logger.info('Creating frozen reference model for KL regularization (quantize={})'.format(
                ref_quantize or 'none'))
        ref_kwargs = dict(trust_remote_code=True)
        if ref_quantize == '4bit':
            ref_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            ref_kwargs['device_map'] = {'': self.device}
        elif ref_quantize == '8bit':
            ref_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            ref_kwargs['device_map'] = {'': self.device}
        else:
            ref_kwargs['torch_dtype'] = torch.bfloat16
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            args.chat_model, **ref_kwargs)
        if not ref_quantize:
            self.ref_model.to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=args.grpo_lr, weight_decay=1e-4)

        if self.use_deepspeed:
            import json as _json
            with open(args.deepspeed_config, 'r') as _f:
                ds_config = _json.load(_f)
            if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
                ds_config["train_micro_batch_size_per_gpu"] = args.grpo_batch_size
            if ds_config.get("gradient_accumulation_steps") == "auto":
                ds_config["gradient_accumulation_steps"] = 1
            if ds_config.get("train_batch_size") == "auto":
                import torch as _torch
                num_gpus = _torch.cuda.device_count()
                ds_config["train_batch_size"] = (
                    ds_config["train_micro_batch_size_per_gpu"]
                    * ds_config["gradient_accumulation_steps"]
                    * num_gpus
                )

            _saved_ds_cfg = args.deepspeed_config
            delattr(args, 'deepspeed_config')

            self.policy_engine, self.optimizer, _, _ = deepspeed.initialize(
                args=args,
                model=self.policy,
                optimizer=self.optimizer,
                config=ds_config,
            )

            args.deepspeed_config = _saved_ds_cfg
            self.policy = self.policy_engine
            self.device = self.policy_engine.device
        else:
            self.policy.to(self.device)

        self.num_samples = args.grpo_num_samples
        self.beta = args.grpo_beta
        self.epochs = args.grpo_epochs
        self.best_metric = None

    def _get_policy_model(self):
        """Unwrap DeepSpeed engine to get the underlying PeftModel."""
        if self.use_deepspeed:
            return self.policy.module
        return self.policy

    def _selective_logits(self, hidden_states, attention_mask, option_token_ids, lm_head_weight):
        """
        Compute logits ONLY for option tokens at the last non-padding position.
        This avoids materializing the full [batch, vocab_size] logit tensor (saves ~4.6GB).
        Returns option_logits: [batch_size, num_options]
        """
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.size(0)
        idx = torch.arange(batch_size, device=self.device)
        last_hidden = hidden_states[idx, seq_lens]
        option_weight = lm_head_weight[option_token_ids]
        option_logits = torch.matmul(
            last_hidden.float(), option_weight.float().T)
        return option_logits

    def _get_option_logprobs(self, input_ids, attention_mask, option_token_ids, num_options):
        """
        Policy forward: get log-probs for option tokens only.
        Returns: log_probs [batch_size, max_options], mask [batch_size, max_options]
        """
        outputs = self.policy(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]

        lm_head_weight = self._get_policy_model().base_model.model.lm_head.weight
        option_logits = self._selective_logits(
            hidden, attention_mask, option_token_ids, lm_head_weight)

        max_opts = option_token_ids.size(0)
        opt_range = torch.arange(max_opts, device=self.device).unsqueeze(0)
        mask = opt_range < num_options.unsqueeze(1)
        option_logits = option_logits.masked_fill(~mask, -1e4)

        log_probs = F.log_softmax(option_logits, dim=-1)
        return log_probs, mask

    @torch.no_grad()
    def _get_ref_logprobs(self, input_ids, attention_mask, option_token_ids, num_options):
        """Get log-probs from the frozen reference model (selective logits)."""
        outputs = self.ref_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]

        lm_head_weight = self.ref_model.lm_head.weight
        option_logits = self._selective_logits(
            hidden, attention_mask, option_token_ids, lm_head_weight)

        max_opts = option_token_ids.size(0)
        opt_range = torch.arange(max_opts, device=self.device).unsqueeze(0)
        mask = opt_range < num_options.unsqueeze(1)
        option_logits = option_logits.masked_fill(~mask, -1e4)

        return F.log_softmax(option_logits, dim=-1)

    def train_step(self, batch: dict) -> dict:
        """One GRPO training step on a batch."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        option_token_ids = batch['option_token_ids'].to(self.device)
        correct_idx = batch['correct_idx'].to(self.device)
        num_options = batch['num_options'].to(self.device)

        batch_size = input_ids.size(0)

        log_probs, mask = self._get_option_logprobs(
            input_ids, attention_mask, option_token_ids, num_options)
        probs = log_probs.exp()

        G = self.num_samples
        sampled_actions = torch.multinomial(
            probs.clamp(min=1e-8), num_samples=G, replacement=True)

        rewards = (sampled_actions == correct_idx.unsqueeze(1)).float()

        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True).clamp(min=1e-4)
        advantages = (rewards - mean_r) / std_r

        sampled_log_probs = log_probs.gather(1, sampled_actions)

        ref_log_probs = self._get_ref_logprobs(
            input_ids, attention_mask, option_token_ids, num_options)
        kl_div = F.kl_div(ref_log_probs, log_probs.exp(), reduction='none', log_target=False)
        kl_div = (kl_div * mask.float()).sum(dim=-1)

        pg_loss = -(advantages.detach() * sampled_log_probs).mean()
        kl_loss = kl_div.mean()
        loss = pg_loss + self.beta * kl_loss

        if self.use_deepspeed:
            self.policy_engine.backward(loss)
            self.policy_engine.step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        greedy_pred = log_probs.argmax(dim=-1)
        greedy_correct = (greedy_pred == correct_idx).float().mean().item()
        sample_correct = rewards.mean().item()

        return {
            'loss': loss.item(),
            'pg_loss': pg_loss.item(),
            'kl_loss': kl_loss.item(),
            'greedy_acc': greedy_correct,
            'sample_acc': sample_correct,
        }

    def _get_transformer_and_lm_head(self):
        """Get the underlying transformer model and lm_head for memory-efficient inference."""
        policy_model = self._get_policy_model()
        causal_lm = policy_model.base_model.model
        transformer = causal_lm.model
        lm_head = causal_lm.lm_head
        return transformer, lm_head

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """Evaluate greedy accuracy on a dataset (memory-efficient).
        
        Uses the underlying transformer model directly to avoid computing
        full vocab logits, which saves significant GPU memory for large models.
        """
        self.policy.eval()
        total_correct = 0
        total_count = 0
        hit1, hit3, hit5, hit10 = 0, 0, 0, 0
        sum_rr = 0.0
        sum_rank = 0

        transformer, lm_head = self._get_transformer_and_lm_head()
        lm_head_weight = lm_head.weight

        for batch in dataloader:
            input_device = next(transformer.parameters()).device
            input_ids = batch['input_ids'].to(input_device)
            attention_mask = batch['attention_mask'].to(input_device)
            option_token_ids = batch['option_token_ids']
            correct_idx = batch['correct_idx']
            num_options = batch['num_options']

            outputs = transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True)
            hidden = outputs.last_hidden_state
            
            head_device = lm_head_weight.device
            option_ids_on_head = option_token_ids.to(head_device)
            option_weight = lm_head_weight[option_ids_on_head].to(hidden.device)
            seq_lens = attention_mask.to(hidden.device).sum(dim=1) - 1
            batch_size = hidden.size(0)
            idx = torch.arange(batch_size, device=hidden.device)
            last_hidden = hidden[idx, seq_lens]
            option_logits = torch.matmul(
                last_hidden.float(), option_weight.float().T)

            num_options = num_options.to(hidden.device)
            correct_idx = correct_idx.to(hidden.device)
            max_opts = option_token_ids.size(0)
            opt_range = torch.arange(max_opts, device=hidden.device).unsqueeze(0)
            opt_mask = opt_range < num_options.unsqueeze(1)
            option_logits = option_logits.masked_fill(~opt_mask, -1e4)

            pred = option_logits.argmax(dim=-1)
            total_correct += (pred == correct_idx).sum().item()

            sorted_indices = option_logits.argsort(dim=-1, descending=True)
            bs = input_ids.size(0)
            for i in range(bs):
                rank_pos = (sorted_indices[i] == correct_idx[i]).nonzero(as_tuple=True)[0].item()
                rank = rank_pos + 1
                sum_rank += rank
                sum_rr += 1.0 / rank
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 5:
                    hit5 += 1
                if rank <= 10:
                    hit10 += 1

            total_count += bs

        self.policy.train()
        n = max(total_count, 1)
        return {
            'accuracy': round(total_correct / n, 4),
            'hit@1': round(hit1 / n, 4),
            'hit@3': round(hit3 / n, 4),
            'hit@5': round(hit5 / n, 4),
            'hit@10': round(hit10 / n, 4),
            'MRR': round(sum_rr / n, 4),
            'MR': round(sum_rank / n, 2),
            'total': total_count,
        }

    def train_loop(self, train_loader, valid_loader=None):
        """Full training loop with GRPO."""
        if self.is_main_process:
            logger.info('=' * 60)
            logger.info('[GRPO] Starting training: {} epochs, {} batches/epoch'.format(
                self.epochs, len(train_loader)))
            logger.info('[GRPO] G={}, beta={}, lr={}'.format(
                self.num_samples, self.beta, self.args.grpo_lr))
            logger.info('=' * 60)

        for epoch in range(self.epochs):
            self.policy.train()
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            losses = AverageMeter('Loss', ':.4f')
            accs = AverageMeter('Acc', ':.4f')

            for step, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                losses.update(metrics['loss'], batch['input_ids'].size(0))
                accs.update(metrics['greedy_acc'], batch['input_ids'].size(0))

                if step % self.args.grpo_print_freq == 0 and self.is_main_process:
                    logger.info(
                        '[GRPO-Train] Epoch {} Step [{}/{}] '
                        'Loss={:.4f} PG={:.4f} KL={:.4f} '
                        'GreedyAcc={:.3f} SampleAcc={:.3f}'.format(
                            epoch, step, len(train_loader),
                            metrics['loss'], metrics['pg_loss'], metrics['kl_loss'],
                            metrics['greedy_acc'], metrics['sample_acc']))

            if self.is_main_process:
                logger.info('[GRPO-Train] Epoch {} summary: Loss={:.4f}, Acc={:.4f}'.format(
                    epoch, losses.avg, accs.avg))

            if valid_loader is not None and self.is_main_process:
                val_metrics = self.evaluate(valid_loader)
                logger.info('[GRPO-Valid] Epoch {}: {}'.format(epoch, json.dumps(val_metrics)))

                is_best = (self.best_metric is None or
                           val_metrics['accuracy'] > self.best_metric['accuracy'])
                if is_best:
                    self.best_metric = val_metrics
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info('[GRPO-Valid] New best model! Acc={}'.format(
                        val_metrics['accuracy']))

            if self.is_main_process:
                self._save_checkpoint(epoch, is_best=False)

        if self.is_main_process:
            if self.best_metric:
                logger.info('[GRPO] Training finished. Best valid metric: {}'.format(
                    json.dumps(self.best_metric)))
            else:
                logger.info('[GRPO] Training finished.')

    def _save_checkpoint(self, epoch, is_best=False):
        save_dir = self.args.grpo_model_dir
        os.makedirs(save_dir, exist_ok=True)

        policy_model = self._get_policy_model()
        if is_best:
            best_dir = os.path.join(save_dir, 'grpo_best')
            policy_model.save_pretrained(best_dir)
            self.tokenizer.save_pretrained(best_dir)
            logger.info(f'Saved best GRPO model to {best_dir}')
        else:
            epoch_dir = os.path.join(save_dir, f'grpo_epoch{epoch}')
            policy_model.save_pretrained(epoch_dir)
            self.tokenizer.save_pretrained(epoch_dir)

    def load_best(self):
        """Load the best checkpoint for evaluation with multi-GPU support."""
        import gc
        best_dir = os.path.join(self.args.grpo_model_dir, 'grpo_best')
        if os.path.exists(best_dir):
            from peft import PeftModel
            logger.info(f'Loading best GRPO model from {best_dir}')
            
            if hasattr(self, 'policy'):
                del self.policy
            if hasattr(self, 'policy_engine'):
                del self.policy_engine
            if hasattr(self, 'ref_model'):
                del self.ref_model
            gc.collect()
            torch.cuda.empty_cache()
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.args.chat_model, trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto')
            new_policy = PeftModel.from_pretrained(base_model, best_dir)
            new_policy.eval()
            
            self.policy = new_policy
            self.device = next(self.policy.parameters()).device
            
            self.use_deepspeed = False
            logger.info(f'Loaded best model with device_map=auto, primary device: {self.device}')
        else:
            logger.warning(f'No best model found at {best_dir}')
