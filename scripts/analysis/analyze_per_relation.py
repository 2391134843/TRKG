#!/usr/bin/env python3
"""
E11: Per-relation performance analysis.

Reads the per-example evaluation JSON files produced by evaluate.py
(forward + backward), groups examples by relation, and computes:
  - MRR, Hits@1, Hits@3, Hits@10
  - Sample count per relation
  - Forward vs Backward comparison

Outputs:
  - analysis_results/e11_per_relation_{TASK}.json   (full per-relation stats)
  - analysis_results/e11_per_relation_{TASK}.csv    (flat table for plotting)
  - analysis_results/e11_per_relation_{TASK}.md     (markdown summary for paper)

Usage:
  python analyze_per_relation.py <TASK> <CHECKPOINT_DIR>
"""
import argparse
import json
import os
import sys
from collections import defaultdict

def load_per_example_json(ckpt_dir: str):
    """Load forward + backward per-example evaluation JSON files.

    Files produced by evaluate.py are at:
      <ckpt_dir>/eval_<split_filename>_{forward,backward}_<ckpt_basename>.json
    e.g. eval_test.txt.json_forward_model_best.mdl.json
    """
    split_filename = 'test.txt.json'
    ckpt_basename = 'model_best.mdl'

    forward_path = os.path.join(
        ckpt_dir, f'eval_{split_filename}_forward_{ckpt_basename}.json')
    backward_path = os.path.join(
        ckpt_dir, f'eval_{split_filename}_backward_{ckpt_basename}.json')

    if not os.path.exists(forward_path):
        raise FileNotFoundError(f'Forward eval JSON not found: {forward_path}')
    if not os.path.exists(backward_path):
        raise FileNotFoundError(f'Backward eval JSON not found: {backward_path}')

    with open(forward_path) as f:
        forward = json.load(f)
    with open(backward_path) as f:
        backward = json.load(f)

    return forward, backward

def compute_relation_metrics(examples):
    """Group by relation and compute MRR, Hits@1/3/10, count."""
    groups = defaultdict(list)
    for ex in examples:
        rel = ex['relation']
        base_rel = rel.replace('inverse ', '') if rel.startswith('inverse ') else rel
        groups[base_rel].append(ex)

    metrics_per_rel = {}
    for rel, exs in groups.items():
        ranks = [e['rank'] for e in exs]
        n = len(ranks)
        mrr = sum(1.0 / r for r in ranks) / n
        h1 = sum(1 for r in ranks if r <= 1) / n
        h3 = sum(1 for r in ranks if r <= 3) / n
        h10 = sum(1 for r in ranks if r <= 10) / n
        mr = sum(ranks) / n
        metrics_per_rel[rel] = {
            'count': n,
            'MRR': round(mrr, 4),
            'Hits@1': round(h1, 4),
            'Hits@3': round(h3, 4),
            'Hits@10': round(h10, 4),
            'MR': round(mr, 2),
        }
    return metrics_per_rel

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

    print(f'[E11] Analyzing {args.task}...')
    print(f'[E11] Checkpoint dir: {args.ckpt_dir}')

    forward, backward = load_per_example_json(args.ckpt_dir)
    print(f'[E11] Forward examples: {len(forward)}, Backward: {len(backward)}')

    combined = forward + backward
    per_rel_combined = compute_relation_metrics(combined)
    per_rel_forward = compute_relation_metrics(forward)
    per_rel_backward = compute_relation_metrics(backward)

    sorted_rels = sorted(per_rel_combined.keys(),
                        key=lambda r: per_rel_combined[r]['count'],
                        reverse=True)

    full_result = {
        'task': args.task,
        'checkpoint_dir': args.ckpt_dir,
        'n_forward': len(forward),
        'n_backward': len(backward),
        'n_total': len(combined),
        'n_relations': len(per_rel_combined),
        'per_relation_combined': per_rel_combined,
        'per_relation_forward': per_rel_forward,
        'per_relation_backward': per_rel_backward,
    }
    json_path = os.path.join(out_dir, f'e11_per_relation_{args.task}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)
    print(f'[E11] Saved JSON: {json_path}')

    csv_path = os.path.join(out_dir, f'e11_per_relation_{args.task}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('relation,count,MRR,Hits@1,Hits@3,Hits@10,MR,'
                'fwd_count,fwd_MRR,fwd_Hits@1,fwd_Hits@10,'
                'bwd_count,bwd_MRR,bwd_Hits@1,bwd_Hits@10\n')
        for rel in sorted_rels:
            m = per_rel_combined[rel]
            fm = per_rel_forward.get(rel, {})
            bm = per_rel_backward.get(rel, {})
            f.write(f'"{rel}",{m["count"]},{m["MRR"]},{m["Hits@1"]},'
                    f'{m["Hits@3"]},{m["Hits@10"]},{m["MR"]},'
                    f'{fm.get("count", 0)},{fm.get("MRR", 0)},'
                    f'{fm.get("Hits@1", 0)},{fm.get("Hits@10", 0)},'
                    f'{bm.get("count", 0)},{bm.get("MRR", 0)},'
                    f'{bm.get("Hits@1", 0)},{bm.get("Hits@10", 0)}\n')
    print(f'[E11] Saved CSV: {csv_path}')

    md_path = os.path.join(out_dir, f'e11_per_relation_{args.task}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# E11 Per-Relation Analysis: {args.task}\n\n')
        f.write(f'- Total test examples: {len(combined)} ')
        f.write(f'({len(forward)} forward + {len(backward)} backward)\n')
        f.write(f'- Number of distinct relations: {len(per_rel_combined)}\n\n')

        f.write('## Overall Per-Relation Metrics (Forward + Backward Combined)\n\n')
        f.write('| Relation | Count | MRR | Hits@1 | Hits@3 | Hits@10 | MR |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        for rel in sorted_rels:
            m = per_rel_combined[rel]
            rel_display = rel if len(rel) < 50 else rel[:47] + '...'
            f.write(f'| {rel_display} | {m["count"]} | {m["MRR"]:.4f} | '
                    f'{m["Hits@1"]:.4f} | {m["Hits@3"]:.4f} | '
                    f'{m["Hits@10"]:.4f} | {m["MR"]:.2f} |\n')

        f.write('\n## Top-5 Best-Performing Relations (by MRR, min 10 samples)\n\n')
        f.write('| Relation | Count | MRR | Hits@1 | Hits@10 |\n')
        f.write('|---|---|---|---|---|\n')
        eligible = [(r, per_rel_combined[r]) for r in sorted_rels
                    if per_rel_combined[r]['count'] >= 10]
        top5 = sorted(eligible, key=lambda x: x[1]['MRR'], reverse=True)[:5]
        for rel, m in top5:
            rel_display = rel if len(rel) < 50 else rel[:47] + '...'
            f.write(f'| {rel_display} | {m["count"]} | {m["MRR"]:.4f} | '
                    f'{m["Hits@1"]:.4f} | {m["Hits@10"]:.4f} |\n')

        f.write('\n## Top-5 Worst-Performing Relations (by MRR, min 10 samples)\n\n')
        f.write('| Relation | Count | MRR | Hits@1 | Hits@10 |\n')
        f.write('|---|---|---|---|---|\n')
        bottom5 = sorted(eligible, key=lambda x: x[1]['MRR'])[:5]
        for rel, m in bottom5:
            rel_display = rel if len(rel) < 50 else rel[:47] + '...'
            f.write(f'| {rel_display} | {m["count"]} | {m["MRR"]:.4f} | '
                    f'{m["Hits@1"]:.4f} | {m["Hits@10"]:.4f} |\n')

    print(f'[E11] Saved Markdown: {md_path}')
    print(f'[E11] Done: {args.task}')

if __name__ == '__main__':
    main()
