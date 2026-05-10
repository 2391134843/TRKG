#!/usr/bin/env python3
"""
E12: PrimeKG per-entity-type performance analysis.

PrimeKG entities have type suffixes in their name like:
  "Aspirin (a drug)" / "hypertension (a disease)" / "BRCA1 (a gene/protein)"
  "Paralysis (a effect/phenotype)" / "Lead (a exposure)"

This script:
  1. Loads per-example eval JSON
  2. Parses head/tail entity types from names
  3. Groups by (head_type, tail_type) combinations
  4. Computes MRR, Hits@1/3/10, sample count for each bucket
  5. Also computes per-type marginal metrics (head-type only, tail-type only)

Outputs:
  - analysis_results/e12_per_entity_type_{TASK}.json
  - analysis_results/e12_per_entity_type_{TASK}.csv
  - analysis_results/e12_per_entity_type_{TASK}.md

Usage:
  python analyze_per_entity_type.py <TASK> <CHECKPOINT_DIR>

Note: Primarily designed for PrimeKG; also works on any dataset where
      entity names have "(a xxx)" suffix.
"""
import argparse
import json
import os
import re
from collections import defaultdict

ENTITY_TYPE_RE = re.compile(r'\(a\s+([^)]+)\)\s*$')

def parse_entity_type(entity_name: str) -> str:
    """Extract type from '... (a drug)' format; returns 'unknown' if not found."""
    m = ENTITY_TYPE_RE.search(entity_name or '')
    if m:
        return m.group(1).strip()
    return 'unknown'

def load_per_example_json(ckpt_dir: str):
    split_filename = 'test.txt.json'
    ckpt_basename = 'model_best.mdl'

    forward_path = os.path.join(
        ckpt_dir, f'eval_{split_filename}_forward_{ckpt_basename}.json')
    backward_path = os.path.join(
        ckpt_dir, f'eval_{split_filename}_backward_{ckpt_basename}.json')

    if not os.path.exists(forward_path) or not os.path.exists(backward_path):
        raise FileNotFoundError(
            f'Missing eval JSON files in: {ckpt_dir}\n'
            f'  Forward: {forward_path}\n'
            f'  Backward: {backward_path}')

    with open(forward_path) as f:
        forward = json.load(f)
    with open(backward_path) as f:
        backward = json.load(f)

    return forward, backward

def aggregate(ranks):
    n = len(ranks)
    if n == 0:
        return None
    return {
        'count': n,
        'MRR': round(sum(1.0 / r for r in ranks) / n, 4),
        'Hits@1': round(sum(1 for r in ranks if r <= 1) / n, 4),
        'Hits@3': round(sum(1 for r in ranks if r <= 3) / n, 4),
        'Hits@10': round(sum(1 for r in ranks if r <= 10) / n, 4),
        'MR': round(sum(ranks) / n, 2),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('ckpt_dir', type=str)
    parser.add_argument('--out-dir', type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'analysis_results'))
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[E12] Analyzing {args.task}...')
    print(f'[E12] Checkpoint dir: {args.ckpt_dir}')

    forward, backward = load_per_example_json(args.ckpt_dir)
    combined = forward + backward
    print(f'[E12] Total examples: {len(combined)}')

    by_combo = defaultdict(list)
    by_head_type = defaultdict(list)
    by_tail_type = defaultdict(list)

    for ex in combined:
        h_type = parse_entity_type(ex.get('head', ''))
        t_type = parse_entity_type(ex.get('tail', ''))
        rank = ex['rank']
        by_combo[(h_type, t_type)].append(rank)
        by_head_type[h_type].append(rank)
        by_tail_type[t_type].append(rank)

    combo_metrics = {f'{h}||{t}': aggregate(ranks)
                     for (h, t), ranks in by_combo.items()}
    head_type_metrics = {h: aggregate(ranks) for h, ranks in by_head_type.items()}
    tail_type_metrics = {t: aggregate(ranks) for t, ranks in by_tail_type.items()}

    all_types = sorted(set(list(by_head_type.keys()) + list(by_tail_type.keys())))

    full = {
        'task': args.task,
        'checkpoint_dir': args.ckpt_dir,
        'n_total': len(combined),
        'entity_types': all_types,
        'combo_metrics': combo_metrics,
        'head_type_metrics': head_type_metrics,
        'tail_type_metrics': tail_type_metrics,
    }
    json_path = os.path.join(out_dir, f'e12_per_entity_type_{args.task}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    print(f'[E12] Saved JSON: {json_path}')

    csv_path = os.path.join(out_dir, f'e12_per_entity_type_{args.task}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('head_type,tail_type,count,MRR,Hits@1,Hits@3,Hits@10,MR\n')
        for (h, t), ranks in sorted(by_combo.items(),
                                     key=lambda x: -len(x[1])):
            m = aggregate(ranks)
            f.write(f'{h},{t},{m["count"]},{m["MRR"]},{m["Hits@1"]},'
                    f'{m["Hits@3"]},{m["Hits@10"]},{m["MR"]}\n')
    print(f'[E12] Saved CSV: {csv_path}')

    md_path = os.path.join(out_dir, f'e12_per_entity_type_{args.task}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# E12 Per-Entity-Type Analysis: {args.task}\n\n')
        f.write(f'- Total test examples: {len(combined)}\n')
        f.write(f'- Distinct entity types: {len(all_types)} ({", ".join(all_types)})\n\n')

        f.write('## Marginal Metrics by Head Entity Type\n\n')
        f.write('| Head Type | Count | MRR | Hits@1 | Hits@3 | Hits@10 | MR |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        for h in sorted(head_type_metrics.keys(),
                        key=lambda x: -head_type_metrics[x]['count']):
            m = head_type_metrics[h]
            f.write(f'| {h} | {m["count"]} | {m["MRR"]:.4f} | '
                    f'{m["Hits@1"]:.4f} | {m["Hits@3"]:.4f} | '
                    f'{m["Hits@10"]:.4f} | {m["MR"]:.2f} |\n')

        f.write('\n## Marginal Metrics by Tail Entity Type\n\n')
        f.write('| Tail Type | Count | MRR | Hits@1 | Hits@3 | Hits@10 | MR |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        for t in sorted(tail_type_metrics.keys(),
                        key=lambda x: -tail_type_metrics[x]['count']):
            m = tail_type_metrics[t]
            f.write(f'| {t} | {m["count"]} | {m["MRR"]:.4f} | '
                    f'{m["Hits@1"]:.4f} | {m["Hits@3"]:.4f} | '
                    f'{m["Hits@10"]:.4f} | {m["MR"]:.2f} |\n')

        f.write('\n## Joint Metrics by (Head Type, Tail Type) — MRR\n\n')
        f.write('|  | ' + ' | '.join(all_types) + ' |\n')
        f.write('|---|' + '|'.join(['---'] * len(all_types)) + '|\n')
        for h in all_types:
            row = [f'**{h}**']
            for t in all_types:
                ranks = by_combo.get((h, t), [])
                if ranks:
                    m = aggregate(ranks)
                    row.append(f'{m["MRR"]:.3f} (n={m["count"]})')
                else:
                    row.append('—')
            f.write('| ' + ' | '.join(row) + ' |\n')

    print(f'[E12] Saved Markdown: {md_path}')
    print(f'[E12] Done: {args.task}')

if __name__ == '__main__':
    main()
