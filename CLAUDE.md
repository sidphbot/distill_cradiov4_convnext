# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowledge distillation framework that trains a lightweight student ConvNeXt model to match the feature representations of a large teacher model (NVIDIA C-RADIOv4-H, ViT-Huge). Two main subsystems: the distillation training engine (`distill/`) and a comprehensive benchmark suite (`bench/`).

## Commands

### Training
```bash
# Full training
python -m distill.launcher --config distill/config.yaml data.image_dir=/path/to/images/

# Debug mode (reduced data caps, frequent eval)
python -m distill.launcher --config distill/config.yaml data.image_dir=/path/to/images/ mode=debug

# CLI dotlist overrides for any config param
python -m distill.launcher --config distill/config.yaml training.lr=3e-4 dataloader.batch_size=32
```

### Hyperparameter Tuning
```bash
python -m distill.tune --config distill/config.yaml --tune-config distill/tune_config.yaml data.image_dir=/path/to/images/
```

### Benchmarking
```bash
# Full suite
python -m bench.run_all --student_ckpt path/to/best.pt --imagenet_root /data/imagenet --coco_root /data/coco --out_dir results/

# Individual stages
python -m bench.rep.eval_alignment --student_ckpt ckpt.pt --dataset imagenet_val --imagenet_root /data/imagenet --out_dir results/
python -m bench.imagenet.extract_features --model student --student_ckpt ckpt.pt --split train --imagenet_root /data/imagenet --out_dir features/
python -m bench.eff.profile --model both --student_ckpt best.pt --out_dir results/
python -m bench.reports.summarize_runs --runs_root /path/to/runs --out_dir comparison/
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Training Pipeline (`distill/`)

- **launcher.py** — Entry point. Loads OmegaConf config, builds models, starts training.
- **model.py** — `DistillModel` wraps student backbone + summary/spatial projection heads. Contains all loss functions: cosine, MSE, Sobel gradient. Also computes 20+ validation metrics (CKA, retrieval, spatial energy, etc.).
- **lightning_module.py** — PyTorch Lightning module with **manual optimization** (disabled automatic). Runs dual-pass per step: clean input for distillation loss + augmented input for consistency loss. Implements gradient accumulation, mixed precision, and dynamic loss weight ramps.
- **data.py** — `ImagePathDataset` returns file paths; `OnlineCollate` loads/resizes images at batch time with 50/50 deterministic pad-or-squash (based on stable hash of path+epoch). `DistillDataModule` handles train/val split via stable hash and multi-source validation (OI, COCO, ImageNet).
- **augment.py** — Albumentations pipeline with strength ramping (alpha-blend between clean and augmented).
- **tune.py** — Optuna integration with TPE sampler, median pruner, and teacher caching across trials.

### Benchmark Suite (`bench/`)

- **run_all.py** — Orchestrates stages: alignment → efficiency → feature extraction → linear probe → kNN → detection train → detection eval → summary.
- **common/** — Shared model loaders (`TeacherBundle`, `StudentBundle`), preprocessing, JSON I/O, timing utilities, constants.
- **rep/eval_alignment.py** — 20+ representation alignment metrics, outputs composite `alignment_score`.
- **imagenet/** — Sharded feature extraction, linear probe, cosine kNN evaluation.
- **coco/** — FasterRCNN head-only training and COCO mAP evaluation.
- **eff/profile.py** — Params, FLOPs (fvcore), latency, throughput.
- **reports/summarize_runs.py** — Aggregates results into CSV and markdown.

## Key Design Decisions

**Config system**: OmegaConf YAML with CLI dotlist overrides. `mode=debug` applies an overlay that reduces data caps and increases eval frequency. All hyperparameters live in config, nothing hardcoded.

**Dual-pass training**: Each step runs the student on both clean and augmented inputs. Clean path produces distillation loss (cosine + MSE + Sobel gradient for both summary and spatial features). Augmented path produces consistency loss (encourages augmentation invariance).

**Dynamic ramps**: Loss weights (`mse_sp`, `grad`, `cons_summary`, `cons_spatial`) and augmentation strength ramp linearly over training with configurable warmup fractions. This stabilizes multi-phase training.

**Online data loading**: Images loaded and transformed at batch time via `OnlineCollate` — no pre-caching. Pad vs squash decision is deterministic per (epoch, path) but varies across epochs.

**Manual Lightning optimization**: Automatic optimization is disabled for explicit gradient accumulation control and immediate graph cleanup to minimize GPU memory.

**Benchmark output convention**: Each script writes `run_meta.json` (git hash, timestamp, CLI args) and `metrics.json` to its output directory.

## No Test Suite

There is no formal test suite. Validation happens through the benchmark suite and debug-mode training runs.
