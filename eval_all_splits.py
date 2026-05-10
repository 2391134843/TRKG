#!/usr/bin/env python3
"""
Evaluate TRKG on train/valid/test splits and save results.
Usage:
  CUDA_VISIBLE_DEVICES=5 python3 eval_all_splits.py \
    --eval-model-path <path_to_best.mdl> \
    --task TCM_KG \
    --train-path data/TCMKG/TCM_KG/train.txt.json \
    --valid-path data/TCMKG/TCM_KG/valid.txt.json \
    --test-path data/TCMKG/TCM_KG/test.txt.json \
    --log-dir log/TRKG/run_1
"""
import os
import sys
import json
import argparse
import torch
import tqdm as tqdm_mod

from time import time
from typing import List, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict

eval_parser = argparse.ArgumentParser(description='Eval TRKG on all splits')
eval_parser.add_argument('--eval-model-path', required=True, type=str)
eval_parser.add_argument('--task', default='TCM_KG', type=str)
eval_parser.add_argument('--train-path', required=True, type=str)
eval_parser.add_argument('--valid-path', required=True, type=str)
eval_parser.add_argument('--test-path', required=True, type=str)
eval_parser.add_argument('--log-dir', default='', type=str)
eval_parser.add_argument('--batch-size', default=512, type=int)
eval_parser.add_argument('--pooling', default='mean', type=str)
eval_parser.add_argument('--max-num-tokens', default=64, type=int)
eval_parser.add_argument('--proj-dim', default=256, type=int)
eval_parser.add_argument('--use-link-graph', action='store_true')
eval_parser.add_argument('--rerank-n-hop', default=2, type=int)
eval_parser.add_argument('--neighbor-weight', default=0.05, type=float)

eval_args = eval_parser.parse_args()

sys.argv = [
    sys.argv[0],
    '--is-test',
    '--task', eval_args.task,
    '--eval-model-path', eval_args.eval_model_path,
    '--train-path', eval_args.train_path,
    '--valid-path', eval_args.valid_path,
    '--pooling', eval_args.pooling,
    '--max-num-tokens', str(eval_args.max_num_tokens),
    '--proj-dim', str(eval_args.proj_dim),
    '--rerank-n-hop', str(eval_args.rerank_n_hop),
    '--neighbor-weight', str(eval_args.neighbor_weight),
    '--batch-size', str(eval_args.batch_size),
]
if eval_args.use_link_graph:
    sys.argv.append('--use-link-graph')

from config import args
from doc import load_data, Example
from predict import BertPredictor
from dict_hub import get_entity_dict, get_all_triplet_dict
from triplet import EntityDict
from rerank import rerank_by_graph
from logger_config import logger

def _setup_entity_dict() -> EntityDict:
    if args.task == 'wiki5m_ind':
        return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()

entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()

@torch.no_grad()
def compute_metrics(hr_tensor, entities_tensor, target, examples, k=3, batch_size=256):
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0

    for start in tqdm_mod.tqdm(range(0, total, batch_size), desc='  Computing metrics'):
        end = start + batch_size
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())
        batch_target = target[start:end]

        rerank_by_graph(batch_score, examples[start:end], entity_dict=entity_dict)

        for idx in range(batch_score.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_score.size(0)
        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1] + 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    return metrics

def eval_single_direction(predictor, entity_tensor, data_path, eval_forward=True, batch_size=256):
    examples = load_data(data_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)
    hr_tensor, _ = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]

    metrics = compute_metrics(
        hr_tensor=hr_tensor, entities_tensor=entity_tensor,
        target=target, examples=examples, batch_size=batch_size)
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('  {} metrics: {}'.format(eval_dir, json.dumps(metrics)))
    return metrics

def eval_split(predictor, entity_tensor, data_path, split_name, batch_size=256):
    logger.info(f'\n{"="*60}')
    logger.info(f'Evaluating on {split_name} set: {data_path}')
    logger.info(f'{"="*60}')

    forward_metrics = eval_single_direction(predictor, entity_tensor, data_path,
                                             eval_forward=True, batch_size=batch_size)
    backward_metrics = eval_single_direction(predictor, entity_tensor, data_path,
                                              eval_forward=False, batch_size=batch_size)
    avg_metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info(f'[{split_name}] Averaged metrics: {json.dumps(avg_metrics)}')
    return {
        'forward': forward_metrics,
        'backward': backward_metrics,
        'average': avg_metrics,
    }

def main():
    logger.info('=' * 60)
    logger.info('TRKG Evaluation on ALL splits')
    logger.info(f'Checkpoint: {eval_args.eval_model_path}')
    logger.info('=' * 60)

    predictor = BertPredictor()
    predictor.load(ckt_path=eval_args.eval_model_path, use_data_parallel=True)

    logger.info('Computing entity embeddings...')
    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)
    logger.info(f'Entity tensor shape: {entity_tensor.shape}')

    all_results = {}

    splits = [
        ('train', eval_args.train_path),
        ('valid', eval_args.valid_path),
        ('test', eval_args.test_path),
    ]

    for split_name, path in splits:
        if not os.path.exists(path):
            logger.warning(f'Skipping {split_name}: {path} not found')
            continue
        result = eval_split(predictor, entity_tensor, path, split_name,
                           batch_size=eval_args.batch_size)
        all_results[split_name] = result

    logger.info('\n' + '=' * 60)
    logger.info('========== FINAL RESULTS SUMMARY ==========')
    logger.info('=' * 60)
    summary_table = []
    for split_name in ['train', 'valid', 'test']:
        if split_name in all_results:
            avg = all_results[split_name]['average']
            logger.info(f'[{split_name:>5}] MRR={avg["mrr"]:.4f}  '
                        f'Hit@1={avg["hit@1"]:.4f}  '
                        f'Hit@3={avg["hit@3"]:.4f}  '
                        f'Hit@10={avg["hit@10"]:.4f}  '
                        f'MR={avg["mean_rank"]:.1f}')
            summary_table.append({'split': split_name, **avg})

    if eval_args.log_dir:
        os.makedirs(eval_args.log_dir, exist_ok=True)
        result_file = os.path.join(eval_args.log_dir, 'TRKG_eval_all_splits.json')
        save_data = {
            'model': 'TRKG',
            'checkpoint': eval_args.eval_model_path,
            'summary': summary_table,
            'detailed_results': all_results,
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        logger.info(f'\nResults saved to: {result_file}')

    logger.info('\n===== Evaluation complete =====')

if __name__ == '__main__':
    main()
