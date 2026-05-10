#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

TRKG_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = TRKG_ROOT / "logs"
STATE_DIR = LOG_DIR / "state"
OUT_MD = LOG_DIR / "results_summary.md"
OUT_CSV = LOG_DIR / "results_summary.csv"
OUT_JSONL = LOG_DIR / "results_summary.jsonl"

TASKS = [
    ("main_tcmkg_stage1",     "Main",      "TCMKG",    "Stage-1 training"),
    ("main_tcmkg_stage2",     "Main",      "TCMKG",    "Stage-2 cand+GRPO (β=0.2)"),
    ("main_primekg_stage1",   "Main",      "PrimeKG",  "Stage-1 training"),
    ("main_primekg_stage2",   "Main",      "PrimeKG",  "Stage-2 cand+GRPO (β=0.2)"),
    ("abl_A1_tcmkg",          "Ablation",  "TCMKG",    "A1 w/o neg-samp (Stage-1)"),
    ("abl_A3_tcmkg",          "Ablation",  "TCMKG",    "A3 w/o link-graph (Stage-1)"),
    ("abl_A4_tcmkg",          "Ablation",  "TCMKG",    "A4 w/o learn-τ (Stage-1)"),
    ("abl_A1_primekg",        "Ablation",  "PrimeKG",  "A1 w/o neg-samp (Stage-1)"),
    ("abl_A3_primekg",        "Ablation",  "PrimeKG",  "A3 w/o link-graph (Stage-1)"),
    ("abl_A4_primekg",        "Ablation",  "PrimeKG",  "A4 w/o learn-τ (Stage-1)"),
    ("abl_B2_tcmkg",          "Ablation",  "TCMKG",    "B2 w/o KL penalty (Stage-2)"),
    ("abl_B4_tcmkg",          "Ablation",  "TCMKG",    "B4 random candidates (Stage-2)"),
    ("abl_B2_primekg",        "Ablation",  "PrimeKG",  "B2 w/o KL penalty (Stage-2)"),
    ("abl_B4_primekg",        "Ablation",  "PrimeKG",  "B4 random candidates (Stage-2)"),
    ("sens_tcmkg_beta_001",   "Sensitivity β", "TCMKG", "β=0.01"),
    ("sens_tcmkg_beta_01",    "Sensitivity β", "TCMKG", "β=0.1"),
    ("sens_tcmkg_beta_05",    "Sensitivity β", "TCMKG", "β=0.5"),
    ("sens_primekg_beta_001", "Sensitivity β", "PrimeKG", "β=0.01"),
    ("sens_primekg_beta_01",  "Sensitivity β", "PrimeKG", "β=0.1"),
    ("sens_primekg_beta_05",  "Sensitivity β", "PrimeKG", "β=0.5"),
    ("sens_tcmkg_topk_05",    "Sensitivity K", "TCMKG", "top-K=5"),
    ("sens_tcmkg_topk_10",    "Sensitivity K", "TCMKG", "top-K=10"),
    ("sens_tcmkg_topk_50",    "Sensitivity K", "TCMKG", "top-K=50"),
    ("sens_primekg_topk_05",  "Sensitivity K", "PrimeKG", "top-K=5"),
    ("sens_primekg_topk_10",  "Sensitivity K", "PrimeKG", "top-K=10"),
    ("sens_primekg_topk_50",  "Sensitivity K", "PrimeKG", "top-K=50"),
]

_RE_TEST_METRICS = re.compile(r"Test metrics:\s*(\{[^}]+\})")
_RE_GRPO_IMPROVE = re.compile(
    r"GRPO improvement: Acc ([\d.]+) -> ([\d.]+) \(\+([\d.]+)\)"
)
_RE_TEST_STAGE1 = re.compile(
    r"Epoch (\d+) summary \| .*? test: MRR=([\d.]+) H@1=([\d.]+) "
    r"H@3=([\d.]+) H@10=([\d.]+) MR=([\d.]+)"
)

def read_text_safe(p: Path) -> str:
    if not p.exists():
        return ""
    try:
        return p.read_text(errors="replace")
    except Exception:
        return ""

def task_state(tid: str) -> str:
    f = STATE_DIR / f"{tid}.state"
    if not f.exists():
        return "pending"
    try:
        return f.read_text().split()[0]
    except Exception:
        return "unknown"

def parse_grpo_metrics(s: str) -> Dict[str, float]:
    m = _RE_TEST_METRICS.search(s)
    if not m:
        return {}
    try:
        d = json.loads(m.group(1))
        return {k: float(v) for k, v in d.items()
                if isinstance(v, (int, float))}
    except Exception:
        return {}

def parse_stage1_final(s: str) -> Dict[str, float]:
    """Return the last Stage-1 epoch's TEST metrics."""
    matches = _RE_TEST_STAGE1.findall(s)
    if not matches:
        return {}
    epoch, mrr, h1, h3, h10, mr = matches[-1]
    return {
        "last_epoch": int(epoch),
        "test_MRR":  float(mrr),
        "test_H@1":  float(h1),
        "test_H@3":  float(h3),
        "test_H@10": float(h10),
        "test_MR":   float(mr),
    }

def gather(tid: str) -> Dict[str, object]:
    main_log = LOG_DIR / f"{tid}.log"
    grpo_log = LOG_DIR / f"{tid}_grpo.log"
    main_text = read_text_safe(main_log)
    grpo_text = read_text_safe(grpo_log)
    merged = main_text + "\n" + grpo_text

    out: Dict[str, object] = {"task": tid, "state": task_state(tid)}

    s1 = parse_stage1_final(main_text)
    if s1:
        out["stage1"] = s1

    grpo_metrics = parse_grpo_metrics(merged)
    if grpo_metrics:
        out["grpo_test"] = grpo_metrics

    m = _RE_GRPO_IMPROVE.search(merged)
    if m:
        out["baseline_acc"] = float(m.group(1))
        out["grpo_acc"] = float(m.group(2))
        out["acc_improvement"] = float(m.group(3))

    return out

def main() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results: List[Dict[str, object]] = []
    for tid, group, dataset, desc in TASKS:
        r = gather(tid)
        r["group"] = group
        r["dataset"] = dataset
        r["desc"] = desc
        results.append(r)

    with open(OUT_JSONL, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    cols = [
        "group", "dataset", "task", "desc", "state",
        "stage1_last_epoch", "stage1_MRR", "stage1_H1", "stage1_H3", "stage1_H10", "stage1_MR",
        "grpo_acc", "grpo_hit1", "grpo_hit3", "grpo_hit10", "grpo_MRR", "grpo_MR",
        "baseline_acc", "acc_improvement",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            row = [
                r.get("group", ""), r.get("dataset", ""), r.get("task", ""),
                r.get("desc", ""), r.get("state", ""),
            ]
            s1 = r.get("stage1", {}) or {}
            row += [s1.get("last_epoch", ""), s1.get("test_MRR", ""),
                    s1.get("test_H@1", ""), s1.get("test_H@3", ""),
                    s1.get("test_H@10", ""), s1.get("test_MR", "")]
            gg = r.get("grpo_test", {}) or {}
            row += [gg.get("accuracy", ""), gg.get("hit@1", ""),
                    gg.get("hit@3", ""), gg.get("hit@10", ""),
                    gg.get("MRR", ""), gg.get("MR", "")]
            row += [r.get("baseline_acc", ""), r.get("acc_improvement", "")]
            w.writerow(row)

    lines: List[str] = []
    lines.append(f"# TRKG-GRPO experiment results\n")
    lines.append(f"_Auto-generated {now}_\n")

    groups = {}
    for r in results:
        groups.setdefault((r["group"], r["dataset"]), []).append(r)

    lines.append("## 1. Main results\n")
    lines.append("| Dataset | Stage | State | Last-epoch (Stage-1) | GRPO test | Δ over baseline |")
    lines.append("|---------|-------|-------|----------------------|-----------|-----------------|")
    for r in results:
        if r["group"] != "Main":
            continue
        s1 = r.get("stage1", {})
        gg = r.get("grpo_test", {})
        s1_txt = ""
        if s1:
            s1_txt = (f"MRR={s1.get('test_MRR', '-'):.3f}"
                      f" H@1={s1.get('test_H@1', '-'):.3f}"
                      f" H@10={s1.get('test_H@10', '-'):.3f}")
        gg_txt = ""
        if gg:
            gg_txt = (f"Acc={gg.get('accuracy', '-')}"
                      f" MRR={gg.get('MRR', '-')}"
                      f" H@1={gg.get('hit@1', '-')}"
                      f" H@10={gg.get('hit@10', '-')}")
        imp = r.get("acc_improvement", "")
        imp_txt = f"+{imp:.4f}" if imp != "" else ""
        stage = "Stage-1" if "stage1" in r["task"] else "Stage-2"
        lines.append(f"| {r['dataset']} | {stage} | {r['state']} | {s1_txt} | {gg_txt} | {imp_txt} |")

    lines.append("\n## 2. Stage-1 ablations (TRKG components)\n")
    lines.append("| Dataset | Config | State | Test MRR | Test H@1 | Test H@3 | Test H@10 | Test MR |")
    lines.append("|---------|--------|-------|----------|----------|----------|-----------|---------|")
    for r in results:
        if r["group"] != "Ablation" or "abl_A" not in r["task"]:
            continue
        s1 = r.get("stage1", {})
        cols = [
            r["dataset"], r["desc"], r["state"],
            f"{s1.get('test_MRR','-'):.3f}"  if s1 else "-",
            f"{s1.get('test_H@1','-'):.3f}"  if s1 else "-",
            f"{s1.get('test_H@3','-'):.3f}"  if s1 else "-",
            f"{s1.get('test_H@10','-'):.3f}" if s1 else "-",
            f"{s1.get('test_MR','-'):.1f}"   if s1 else "-",
        ]
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")

    lines.append("\n## 3. Stage-2 ablations (GRPO components)\n")
    lines.append("| Dataset | Config | State | GRPO Acc | GRPO MRR | GRPO H@1 | GRPO H@10 |")
    lines.append("|---------|--------|-------|----------|----------|----------|-----------|")
    for r in results:
        if r["group"] != "Ablation" or "abl_B" not in r["task"]:
            continue
        gg = r.get("grpo_test", {})
        cols = [
            r["dataset"], r["desc"], r["state"],
            f"{gg.get('accuracy','-')}" if gg else "-",
            f"{gg.get('MRR','-')}"      if gg else "-",
            f"{gg.get('hit@1','-')}"    if gg else "-",
            f"{gg.get('hit@10','-')}"   if gg else "-",
        ]
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")

    lines.append("\n## 4. Sensitivity: KL penalty β\n")
    lines.append("| Dataset | β | State | GRPO Acc | GRPO MRR | GRPO H@1 | GRPO H@10 |")
    lines.append("|---------|---|-------|----------|----------|----------|-----------|")
    for r in results:
        if r["group"] != "Sensitivity β":
            continue
        gg = r.get("grpo_test", {})
        cols = [
            r["dataset"], r["desc"], r["state"],
            f"{gg.get('accuracy','-')}" if gg else "-",
            f"{gg.get('MRR','-')}"      if gg else "-",
            f"{gg.get('hit@1','-')}"    if gg else "-",
            f"{gg.get('hit@10','-')}"   if gg else "-",
        ]
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")

    lines.append("\n## 5. Sensitivity: Stage-1 top-K\n")
    lines.append("| Dataset | K | State | GRPO Acc | GRPO MRR | GRPO H@1 | GRPO H@10 |")
    lines.append("|---------|---|-------|----------|----------|----------|-----------|")
    for r in results:
        if r["group"] != "Sensitivity K":
            continue
        gg = r.get("grpo_test", {})
        cols = [
            r["dataset"], r["desc"], r["state"],
            f"{gg.get('accuracy','-')}" if gg else "-",
            f"{gg.get('MRR','-')}"      if gg else "-",
            f"{gg.get('hit@1','-')}"    if gg else "-",
            f"{gg.get('hit@10','-')}"   if gg else "-",
        ]
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")

    lines.append("")
    lines.append(f"_CSV: `logs/results_summary.csv`   JSONL: `logs/results_summary.jsonl`_")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_JSONL}")

if __name__ == "__main__":
    main()
