# TRKG: Two-Stage LLM Retrieval and Reinforcement Reranking for Biomedical Knowledge Graph Completion

![Python](https://img.shields.io/badge/python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-red)

This repository provides the official PyTorch implementation for the paper:
**TRKG: Two-Stage LLM Retrieval and Reinforcement Reranking for Biomedical Knowledge Graph Completion**.

## Installation

1. Clone this git repository:

   ```bash
   git clone <repo-url>
   cd TRKG
   ```

2. Create a new conda environment:

   ```bash
   conda create --name trkg python=3.10
   conda activate trkg
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Models default to HuggingFace IDs (e.g. `Qwen/Qwen2.5-7B`,
   etc.) and are downloaded automatically on first use. To use a local copy
   instead, export `BASE_MODEL` / `CHAT_MODEL` (or `BASE_MODEL_7B` /
   `CHAT_MODEL_7B` for the 7B variant) before running any script.

## Data

Datasets are bundled in `data/` and ready to use:

- **WN18RR** — WordNet lexical relations (40,943 entities, 11 relations)
- **PrimeKG** — biomedical knowledge graph (10,344 entities, 11 relations)
- **TCMKG** — traditional Chinese medicine knowledge graph (TCM herbs,
  symptoms, prescriptions)

Each dataset directory contains `entities.json`, `relations.json` and
`train.txt.json` / `valid.txt.json` / `test.txt.json` with denormalised
triples carrying `head_desc` / `tail_desc` for the contrastive encoder.

## Run the models

### Stage 1 + Stage 2 with the 0.5B backbone

```bash
# WN18RR — Stage 1 + Stage 2 in one shot
bash scripts/WN18RR/run_pipeline_05B.sh

# PrimeKG
bash scripts/main/primekg_stage1.sh
bash scripts/main/primekg_stage2.sh   # waits for Stage 1 to finish

# TCMKG
bash scripts/main/tcmkg_stage1.sh
bash scripts/main/tcmkg_stage2.sh
```

### Stage 1 + Stage 2 with the 7B backbone (DeepSpeed)

```bash
bash scripts/main7b/wn18rr_7b.sh
bash scripts/main7b/primekg_7b.sh
bash scripts/main7b/tcmkg_7b.sh
```

### Ablations and sensitivity studies

```bash
# Stage 1 ablations: drop pre-batch + self-negative / link-graph / learnable t
bash scripts/abl_primekg/s1_a1_no_negsamp.sh
bash scripts/abl_primekg/s1_a2_no_linkgraph.sh
bash scripts/abl_primekg/s1_a3_no_learnt.sh

# Stage 2 ablations: zero KL / random candidates
bash scripts/abl_primekg/s2_b1_beta0.sh
bash scripts/abl_primekg/s2_b2_random_cand.sh

# Sensitivity over GRPO beta and Stage-2 top-k
bash scripts/sensitivity/primekg_beta_001.sh
bash scripts/sensitivity/primekg_topk_10.sh
```

The same set of `s1_*.sh` / `s2_*.sh` and `*_beta_*.sh` / `*_topk_*.sh`
scripts is mirrored under `scripts/abl_tcm/` and `scripts/sensitivity/`
for the TCMKG dataset.

You can adjust hyperparameters by editing the corresponding shell script
or by exporting environment overrides (e.g. `S1_LR`, `S2_GRPO_BETA`,
`S2_MAX_CAND`, `MAIN7B_GPUS`) before invoking it.

## Project Structure

```
TRKG/
├── main.py                     # Stage 1 (TRKG contrastive) entry point
├── run_grpo_rerank.py          # Stage 2 (GRPO chat-rerank) entry point
├── generate_candidates.py      # Stage 1 -> Stage 2 bridge: top-K candidates
├── trainer.py                  # Stage 1 trainer (contrastive + bidirectional)
├── grpo_trainer.py             # Stage 2 trainer (GRPO + KL regularisation)
├── models.py                   # TRKG model (shared encoder + projection heads)
├── chat_rerank_dataset.py      # Multiple-choice prompt dataset for GRPO
├── doc.py / triplet.py         # Triple loading + link graph + masking
├── evaluate.py / eval_all_splits.py / metric.py
│                               # Evaluation (Hits@k, MRR, MR)
├── config.py                   # Stage 1 argparse definitions
├── ds_config_stage{1,2}*.json  # DeepSpeed configs (0.5B / 7B)
│
├── data/                       # WN18RR, PrimeKG, TCMKG (entities + triples)
│
├── scripts/
│   ├── common.sh               # Shared paths, env activation, helpers
│   ├── main/                   # Main 0.5B Stage 1 / Stage 2 per dataset
│   ├── main7b/                 # Main 7B end-to-end pipeline per dataset
│   ├── WN18RR/                 # 0.5B / 7B pipeline scripts for WN18RR
│   ├── abl_primekg/            # Stage-1 (s1_a*) + Stage-2 (s2_b*) ablations
│   ├── abl_tcm/                # Same ablation set on TCMKG
│   ├── sensitivity/            # GRPO beta and top-k sensitivity studies
│   ├── analysis/               # Per-relation / per-entity-type analyses
│   └── summarize_results.py    # Summarise metrics across runs
│
├── requirements.txt
└── README.md
```

After training, checkpoints land in `checkpoint/`, top-K candidates in
`candidates/`, and stdout/stderr logs in `logs/` (these directories are
created on demand and ignored by git).

## Evaluation Metrics

All metrics are computed under the **filtered** setting (known correct
triples in train + valid are excluded from ranking):

- **MRR** — mean reciprocal rank
- **Hits@1 / @3 / @10** — proportion of correct entities ranked in the top *k*
- **MR** — mean rank

## Citation

If you find this work useful, please cite our paper.

## Acknowledgement

This work builds on the following excellent contributions.

**Datasets and benchmarks**

- **WN18RR** — Convolutional 2D Knowledge Graph Embeddings (Dettmers et al.,
  AAAI 2018)
- **PrimeKG** — Building a Knowledge Graph to Enable Precision Medicine
  (Chandak et al., *Scientific Data* 2023)

**Embedding-based KGC baselines**

- **TransE** — Translating Embeddings for Modeling Multi-relational Data
  (Bordes et al., NeurIPS 2013)
- **DistMult** — Embedding Entities and Relations for Learning and Inference
  in Knowledge Bases (Yang et al., ICLR 2015)
- **ComplEx** — Complex Embeddings for Simple Link Prediction (Trouillon
  et al., ICML 2016)
- **RotatE** — Knowledge Graph Embedding by Relational Rotation in Complex
  Space (Sun et al., ICLR 2019)
- **UniGE** — Bridging the Space Gap: Unifying Geometry Knowledge Graph
  Embedding with Optimal Transport (Liu et al., WWW 2024)
- **MGTCA** — Mixed Geometry Message and Trainable Convolutional Attention
  Network for Knowledge Graph Completion (Shang et al., AAAI 2024)
- **MRME** — Multi-view Riemannian Manifolds Fusion Enhancement for Knowledge
  Graph Completion (Li et al., *IEEE TKDE* 2025)

**Language-model-based KGC baselines and related work**

- **KG-BERT** — BERT for Knowledge Graph Completion (Yao et al., 2019)
- **SimKGC** — Simple Contrastive Knowledge Graph Completion with Pre-trained
  Language Models (Wang et al., ACL 2022)
- **KICGPT** — Large Language Model with Knowledge in Context for Knowledge
  Graph Completion (Wei et al., EMNLP Findings 2023)
- **BioGraphFusion** — Graph Knowledge Embedding for Biological Completion
  and Reasoning (Lin et al., *Bioinformatics* 2025)

**Foundation models, training and PEFT**

- **Qwen2.5** — Qwen2.5 Technical Report (Alibaba, 2024)
- **GRPO** — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
  Open Language Models (Shao et al., 2024)
- **PEFT / LoRA** — Low-Rank Adaptation of Large Language Models (Hu et al.,
  ICLR 2022)
- **DeepSpeed** — ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models (Rajbhandari et al., SC 2020)

## 🤝 Contact

If you have any questions, please feel free to open an issue or contact
us at linyuli@stu.pku.edu.cn.
