import os
import random
import torch
import argparse
import warnings

import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='TRKG arguments')
parser.add_argument('--pretrained-model', default='Qwen/Qwen2.5-0.5B', type=str,
                    help='HuggingFace model id or local path to the pretrained LLM')
parser.add_argument('--task', default='FB15k237', type=str,
                    help='dataset name')
parser.add_argument('--train-path', default='', type=str,
                    help='path to training data')
parser.add_argument('--valid-path', default='', type=str,
                    help='path to valid data')
parser.add_argument('--test-path', default='', type=str,
                    help='path to test data')
parser.add_argument('--model-dir', default='', type=str,
                    help='path to model dir')
parser.add_argument('--warmup', default=400, type=int,
                    help='warmup steps')
parser.add_argument('--max-to-keep', default=3, type=int,
                    help='max number of checkpoints to keep')
parser.add_argument('--grad-clip', default=10.0, type=float,
                    help='gradient clipping')
parser.add_argument('--pooling', default='mean', type=str,
                    help='pooling strategy: mean, last, max')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='dropout on final linear layer')
parser.add_argument('--use-amp', action='store_true',
                    help='Use amp if available')
parser.add_argument('--t', default=0.05, type=float,
                    help='temperature parameter')
parser.add_argument('--use-link-graph', action='store_true',
                    help='use neighbors from link graph as context')
parser.add_argument('--eval-every-n-step', default=5000, type=int,
                    help='evaluate every n steps')
parser.add_argument('--pre-batch', default=0, type=int,
                    help='number of pre-batch used for negatives')
parser.add_argument('--pre-batch-weight', default=0.5, type=float,
                    help='the weight for logits from pre-batch negatives')
parser.add_argument('--additive-margin', default=0.02, type=float,
                    help='additive margin for InfoNCE loss function')
parser.add_argument('--finetune-t', action='store_true',
                    help='make temperature a trainable parameter')
parser.add_argument('--max-num-tokens', default=64, type=int,
                    help='maximum number of tokens')
parser.add_argument('--use-self-negative', action='store_true',
                    help='use head entity as negative')

parser.add_argument('--lora-r', default=16, type=int,
                    help='LoRA rank')
parser.add_argument('--lora-alpha', default=32, type=int,
                    help='LoRA alpha')
parser.add_argument('--lora-dropout', default=0.05, type=float,
                    help='LoRA dropout')

parser.add_argument('--proj-dim', default=256, type=int,
                    help='projection dimension for contrastive learning')

parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='cosine', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    help='print frequency')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training')
parser.add_argument('--gradient-accumulation-steps', default=1, type=int,
                    help='gradient accumulation steps')

parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training (set by deepspeed)')
parser.add_argument('--deepspeed', default='', type=str,
                    help='path to deepspeed config json')

parser.add_argument('--is-test', action='store_true',
                    help='is in test mode or not')
parser.add_argument('--rerank-n-hop', default=2, type=int,
                    help='use n-hops node for re-ranking')
parser.add_argument('--neighbor-weight', default=0.05, type=float,
                    help='weight for re-ranking entities')
parser.add_argument('--eval-model-path', default='', type=str,
                    help='path to model for evaluation')

args, _ = parser.parse_known_args()

assert args.pooling in ['mean', 'last', 'max']
assert args.lr_scheduler in ['linear', 'cosine']

if args.model_dir:
    os.makedirs(args.model_dir, exist_ok=True)
elif args.eval_model_path:
    assert os.path.exists(args.eval_model_path)
    args.model_dir = os.path.dirname(args.eval_model_path)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available')

if not torch.cuda.is_available():
    args.use_amp = False
    args.print_freq = 1
    warnings.warn('GPU is not available')
