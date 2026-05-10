"""
Generate random candidates for B4 ablation experiment.
Replaces Stage1-retrieved candidates with randomly sampled entities,
validating the importance of Stage1 recall quality.
"""
import json
import random
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='Generate random candidates for Stage2 ablation (B4)')
    parser.add_argument('--src-dir', required=True,
                        help='Source candidates directory (from Stage1)')
    parser.add_argument('--entities-file', required=True,
                        help='Path to entities.json for the dataset')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for random candidates')
    parser.add_argument('--num-candidates', type=int, default=20,
                        help='Number of random candidates per example')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.entities_file, 'r', encoding='utf-8') as f:
        all_entities = json.load(f)

    entity_map = {e['entity_id']: e for e in all_entities}
    entity_ids = list(entity_map.keys())
    entity_id_set = set(entity_ids)
    print(f'Loaded {len(entity_ids)} entities from {args.entities_file}')

    for split in ['train', 'valid', 'test']:
        src_file = os.path.join(args.src_dir, f'candidates_{split}.json')
        if not os.path.exists(src_file):
            print(f'Skipping {split}: {src_file} not found')
            continue

        print(f'[{split}] Loading {src_file} ...')
        with open(src_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f'[{split}] Loaded {len(data)} examples, generating random candidates...')

        for idx, item in enumerate(data):
            tail_id = item['tail_id']

            gt_candidate = None
            for c in item['candidates']:
                if c['is_correct']:
                    gt_candidate = {
                        'entity_id': c['entity_id'], 'entity': c['entity'],
                        'entity_desc': c.get('entity_desc', ''),
                        'score': 0.0, 'rank': 0, 'is_correct': True,
                    }
                    break

            if gt_candidate is None:
                ent = entity_map.get(tail_id)
                if ent:
                    gt_candidate = {
                        'entity_id': ent['entity_id'], 'entity': ent['entity'],
                        'entity_desc': ent.get('entity_desc', ''),
                        'score': 0.0, 'rank': 0, 'is_correct': True,
                    }
                else:
                    gt_candidate = {
                        'entity_id': tail_id, 'entity': item['tail'],
                        'entity_desc': '',
                        'score': 0.0, 'rank': 0, 'is_correct': True,
                    }

            n_random = min(args.num_candidates - 1, len(entity_ids) - 1)
            sampled_ids = random.sample(entity_ids, n_random + 1)
            sampled_ids = [eid for eid in sampled_ids if eid != tail_id][:n_random]

            random_candidates = []
            for eid in sampled_ids:
                e = entity_map[eid]
                random_candidates.append({
                    'entity_id': e['entity_id'], 'entity': e['entity'],
                    'entity_desc': e.get('entity_desc', ''),
                    'score': 0.0, 'rank': 0, 'is_correct': False,
                })

            insert_pos = random.randint(0, len(random_candidates))
            random_candidates.insert(insert_pos, gt_candidate)
            for i, c in enumerate(random_candidates):
                c['rank'] = i

            item['candidates'] = random_candidates

            if (idx + 1) % 10000 == 0:
                print(f'[{split}] Processed {idx + 1}/{len(data)} examples')

        out_file = os.path.join(args.output_dir, f'candidates_{split}.json')
        print(f'[{split}] Writing {len(data)} examples to {out_file} ...')
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f'[{split}] Done.')


if __name__ == '__main__':
    main()
