"""
Stage 1: Generate top-K candidates using a trained TRKG model.
Saves candidate lists as JSON for GRPO reranking training.
"""
import os
import json
import tqdm
import torch
import argparse

from time import time
from typing import List

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

def generate_for_split(predictor,
                       entity_dict: EntityDict,
                       all_triplet_dict,
                       entity_tensor: torch.tensor,
                       data_path: str,
                       split_name: str,
                       top_k: int = 20,
                       output_dir: str = './candidates'):
    """Generate top-K candidates for one data split (forward + backward)."""
    logger.info(f'Generating top-{top_k} candidates for {split_name} from {data_path}')
    logger.info(f'In test mode: {args.is_test}')

    examples = load_data(data_path, add_forward_triplet=True, add_backward_triplet=True)

    hr_tensor, _ = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)

    entity_id_list = [ex.entity_id for ex in entity_dict.entity_exs]

    hr_norm = hr_tensor / hr_tensor.norm(dim=1, keepdim=True).clamp(min=1e-8)
    ent_norm = entity_tensor / entity_tensor.norm(dim=1, keepdim=True).clamp(min=1e-8)

    results = []
    batch_size = 256
    total = hr_norm.size(0)
    for start in tqdm.tqdm(range(0, total, batch_size), desc=split_name):
        end = min(start + batch_size, total)
        scores = torch.mm(hr_norm[start:end], ent_norm.t())
        topk_scores, topk_indices = scores.topk(top_k, dim=1)

        for i in range(end - start):
            ex = examples[start + i]
            head_id = ex.head_id
            tail_id = ex.tail_id
            direction = 'forward' if ex.relation and not ex.relation.startswith('inverse ') else 'backward'

            candidates = []
            gt_in_topk = False
            for rank_j in range(top_k):
                eidx = topk_indices[i, rank_j].item()
                ent_id = entity_id_list[eidx]
                ent_ex = entity_dict.get_entity_by_idx(eidx)
                is_correct = (ent_id == tail_id)
                if is_correct:
                    gt_in_topk = True
                candidates.append({
                    'entity_id': ent_id,
                    'entity': ent_ex.entity,
                    'entity_desc': ent_ex.entity_desc,
                    'score': round(topk_scores[i, rank_j].item(), 6),
                    'rank': rank_j,
                    'is_correct': is_correct,
                })

            if not gt_in_topk:
                gt_eidx = entity_dict.entity2idx.get(tail_id, None)
                if gt_eidx is not None:
                    ent_ex = entity_dict.get_entity_by_idx(gt_eidx)
                    candidates[-1] = {
                        'entity_id': tail_id,
                        'entity': ent_ex.entity,
                        'entity_desc': ent_ex.entity_desc,
                        'score': 0.0,
                        'rank': top_k - 1,
                        'is_correct': True,
                    }

            results.append({
                'head_id': head_id,
                'head': ex.head,
                'head_desc': getattr(ex, 'head_desc', ''),
                'relation': ex.relation,
                'tail_id': tail_id,
                'tail': ex.tail,
                'candidates': candidates,
                'direction': direction,
            })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'candidates_{split_name}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f'Saved {len(results)} examples to {out_path}')
    return results

def main():
    custom_parser = argparse.ArgumentParser()
    custom_parser.add_argument('--top-k', type=int, default=20)
    custom_parser.add_argument('--output-dir', type=str, default='./candidates')
    custom_parser.add_argument('--splits', type=str, default='train,valid,test',
                               help='comma-separated: train,valid,test')
    custom_args, _ = custom_parser.parse_known_args()

    top_k = custom_args.top_k
    output_dir = custom_args.output_dir
    splits = [s.strip() for s in custom_args.splits.split(',')]

    logger.info(json.dumps(vars(args), ensure_ascii=False, indent=2))

    entity_dict = _setup_entity_dict()
    all_triplet_dict = get_all_triplet_dict()

    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path, use_data_parallel=True)
    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    split_path_map = {
        'train': args.train_path,
        'valid': args.valid_path,
        'test': getattr(args, 'test_path', ''),
    }

    t_start = time()
    for split in splits:
        path = split_path_map.get(split, '')
        if not path or not os.path.exists(path):
            logger.warning(f'Skipping split {split}: path not found ({path})')
            continue
        generate_for_split(
            predictor=predictor,
            entity_dict=entity_dict,
            all_triplet_dict=all_triplet_dict,
            entity_tensor=entity_tensor,
            data_path=path,
            split_name=split,
            top_k=top_k,
            output_dir=output_dir)

    logger.info(f'All candidates generated in {time() - t_start:.1f}s')

if __name__ == '__main__':
    main()
