"""
Main entry point for GRPO-based Chat Reranking pipeline.

Usage:
  python run_grpo_rerank.py \
    --chat-model Qwen/Qwen2.5-0.5B-Instruct \
    --candidates-dir ./candidates/WN18RR \
    --task WN18RR \
    --grpo-model-dir ./checkpoint/grpo_WN18RR \
    --grpo-epochs 3 --grpo-lr 5e-5 --grpo-batch-size 16 \
    --grpo-num-samples 8 --grpo-beta 0.1 \
    --max-candidates 10 --mode train_and_eval
"""
import os
import json
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from functools import partial

from transformers import AutoModelForCausalLM
from chat_rerank_dataset import ChatRerankDataset, collate_chat
from grpo_trainer import GRPOTrainer
from logger_config import logger

def parse_args():
    parser = argparse.ArgumentParser(description='GRPO Chat Reranking for KGC')

    parser.add_argument('--chat-model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='HuggingFace model id or local path to the instruction-tuned LLM')

    parser.add_argument('--candidates-dir', type=str, required=True,
                        help='directory containing candidates_{split}.json files')
    parser.add_argument('--task', type=str, default='WN18RR',
                        help='dataset name (WN18RR or FB15k237)')
    parser.add_argument('--max-candidates', type=int, default=5,
                        help='max number of candidate options per prompt')

    parser.add_argument('--grpo-model-dir', type=str, default='./checkpoint/grpo',
                        help='directory to save GRPO model checkpoints')
    parser.add_argument('--grpo-epochs', type=int, default=5)
    parser.add_argument('--grpo-lr', type=float, default=2e-5)
    parser.add_argument('--grpo-batch-size', type=int, default=8)
    parser.add_argument('--grpo-num-samples', type=int, default=16,
                        help='G: number of samples per prompt for GRPO')
    parser.add_argument('--grpo-beta', type=float, default=0.2,
                        help='KL penalty weight')
    parser.add_argument('--grpo-lora-r', type=int, default=16)
    parser.add_argument('--grpo-lora-alpha', type=int, default=32)
    parser.add_argument('--grpo-print-freq', type=int, default=20)
    parser.add_argument('--grpo-max-length', type=int, default=512,
                        help='tokenizer max_length for chat prompts; raise '
                             'when --max-candidates is large (e.g. 2048 for '
                             'max-candidates=100)')

    parser.add_argument('--mode', type=str, default='train_and_eval',
                        choices=['train_and_eval', 'eval_only'],
                        help='train_and_eval or eval_only')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--ref-model-quantize', type=str, default='',
                        choices=['', '4bit', '8bit'],
                        help='quantize reference model to save GPU memory (for large models)')

    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank for distributed training (set by deepspeed)')
    parser.add_argument('--deepspeed_config', type=str, default='',
                        help='path to deepspeed config json for GRPO stage')

    return parser.parse_args()

def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    use_deepspeed = bool(args.deepspeed_config)
    is_main = (args.local_rank <= 0)

    if use_deepspeed and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    if is_main:
        logger.info('GRPO Reranking args: {}'.format(
            json.dumps(vars(args), ensure_ascii=False, indent=2)))

    trainer = GRPOTrainer(args)
    tokenizer = trainer.tokenizer

    collate_fn = partial(collate_chat, tokenizer=tokenizer,
                          max_length=args.grpo_max_length)

    train_path = os.path.join(args.candidates_dir, 'candidates_train.json')
    valid_path = os.path.join(args.candidates_dir, 'candidates_valid.json')
    test_path = os.path.join(args.candidates_dir, 'candidates_test.json')

    train_loader, valid_loader, test_loader = None, None, None

    if args.mode == 'train_and_eval':
        if not os.path.exists(train_path) and os.path.exists(valid_path):
            if is_main:
                logger.warning(f'Train candidates not found, falling back to valid: {valid_path}')
            train_path = valid_path
        assert os.path.exists(train_path), f'Train candidates not found: {train_path}'
        train_ds = ChatRerankDataset(train_path, task=args.task,
                                     max_candidates=args.max_candidates,
                                     shuffle_candidates=True)
        if use_deepspeed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.grpo_batch_size,
            shuffle=(train_sampler is None), sampler=train_sampler,
            collate_fn=collate_fn, num_workers=args.workers, drop_last=True)

    if os.path.exists(valid_path):
        valid_ds = ChatRerankDataset(valid_path, task=args.task,
                                     max_candidates=args.max_candidates,
                                     shuffle_candidates=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=args.grpo_batch_size * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=args.workers)

    if os.path.exists(test_path):
        test_ds = ChatRerankDataset(test_path, task=args.task,
                                    max_candidates=args.max_candidates,
                                    shuffle_candidates=False)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.grpo_batch_size * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=args.workers)

    if args.mode == 'train_and_eval' and train_loader:
        trainer.train_loop(train_loader, valid_loader)

    if test_loader and is_main:
        import gc
        
        if args.mode == 'eval_only' or trainer.best_metric is not None:
            trainer.load_best()

        logger.info('=' * 60)
        logger.info('[GRPO-Test] Evaluating on test set...')
        logger.info('=' * 60)
        test_metrics = trainer.evaluate(test_loader)
        logger.info('[GRPO-Test] Test metrics: {}'.format(json.dumps(test_metrics)))

        logger.info('[GRPO-Test] Freeing GRPO model memory for baseline evaluation...')
        if hasattr(trainer, 'policy'):
            del trainer.policy
        if hasattr(trainer, 'policy_engine'):
            del trainer.policy_engine
        gc.collect()
        torch.cuda.empty_cache()

        logger.info('[GRPO-Test] Evaluating baseline (raw Instruct model) on test...')
        baseline_eval = GRPOBaselineEvaluator(args)
        baseline_metrics = baseline_eval.evaluate(test_loader)
        logger.info('[GRPO-Test] Baseline metrics: {}'.format(json.dumps(baseline_metrics)))

        logger.info('=' * 60)
        logger.info('[GRPO-Test] GRPO improvement: Acc {:.4f} -> {:.4f} (+{:.4f})'.format(
            baseline_metrics['accuracy'], test_metrics['accuracy'],
            test_metrics['accuracy'] - baseline_metrics['accuracy']))
        logger.info('=' * 60)

        def _fmt(m):
            if not m:
                return '(none)'
            return (
                f'acc={m.get("accuracy", 0):.4f}  hit@1={m.get("hit@1", 0):.4f}  '
                f'hit@3={m.get("hit@3", 0):.4f}  hit@5={m.get("hit@5", 0):.4f}  '
                f'hit@10={m.get("hit@10", 0):.4f}  MRR={m.get("MRR", 0):.4f}  '
                f'MR={m.get("MR", 0):.2f}  total={m.get("total", 0)}'
            )
        banner = '=' * 78
        logger.info(banner)
        logger.info('[FINAL SUMMARY] task=%s  grpo_model_dir=%s', args.task, args.grpo_model_dir)
        logger.info('[FINAL SUMMARY] GRPO     test : %s', _fmt(test_metrics))
        logger.info('[FINAL SUMMARY] BASELINE test: %s', _fmt(baseline_metrics))
        logger.info(banner)

        try:
            import os as _os
            _os.makedirs(args.grpo_model_dir, exist_ok=True)
            with open(_os.path.join(args.grpo_model_dir, 'final_metrics.json'), 'w') as f:
                json.dump({
                    'task': args.task,
                    'grpo_model_dir': args.grpo_model_dir,
                    'max_candidates': args.max_candidates,
                    'grpo_test_metrics': test_metrics,
                    'baseline_test_metrics': baseline_metrics,
                    'delta_accuracy': round(
                        test_metrics['accuracy'] - baseline_metrics['accuracy'], 4),
                }, f, indent=2)
            logger.info('[FINAL SUMMARY] metrics saved to %s/final_metrics.json',
                        args.grpo_model_dir)
        except Exception as e:
            logger.warning('[FINAL SUMMARY] failed to save final_metrics.json: %s', e)
    elif is_main:
        logger.info('No test candidates found, skipping test evaluation')

class GRPOBaselineEvaluator:
    """Evaluates the raw Instruct model (without GRPO) as a baseline, memory-efficient.
    
    Uses device_map='auto' to distribute large models across multiple GPUs,
    and calls transformer directly (without lm_head) to avoid full vocab logits.
    """

    def __init__(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.chat_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map='auto')
        self.model.eval()
        self.transformer = self.model.model
        self.lm_head = self.model.lm_head
        self.device = next(self.transformer.parameters()).device

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        total_correct = 0
        total_count = 0
        hit1, hit3, hit5, hit10 = 0, 0, 0, 0
        sum_rr = 0.0
        sum_rank = 0

        lm_head_weight = self.lm_head.weight

        for batch in dataloader:
            input_device = self.device
            input_ids = batch['input_ids'].to(input_device)
            attention_mask = batch['attention_mask'].to(input_device)
            option_token_ids = batch['option_token_ids']
            correct_idx = batch['correct_idx']
            num_options = batch['num_options']

            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True)
            hidden = outputs.last_hidden_state
            
            seq_lens = attention_mask.to(hidden.device).sum(dim=1) - 1
            bs = input_ids.size(0)
            idx = torch.arange(bs, device=hidden.device)
            last_hidden = hidden[idx, seq_lens]

            head_device = lm_head_weight.device
            option_ids_on_head = option_token_ids.to(head_device)
            option_weight = lm_head_weight[option_ids_on_head].to(hidden.device)
            option_logits = torch.matmul(last_hidden.float(), option_weight.float().T)

            num_options = num_options.to(hidden.device)
            correct_idx = correct_idx.to(hidden.device)
            max_opts = option_token_ids.size(0)
            opt_range = torch.arange(max_opts, device=hidden.device).unsqueeze(0)
            mask = opt_range < num_options.unsqueeze(1)
            option_logits = option_logits.masked_fill(~mask, -1e4)

            pred = option_logits.argmax(dim=-1)
            total_correct += (pred == correct_idx).sum().item()

            sorted_indices = option_logits.argsort(dim=-1, descending=True)
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

if __name__ == '__main__':
    main()
