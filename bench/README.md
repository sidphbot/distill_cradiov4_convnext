# Benchmark Suite for Distilled Student Models

Evaluate distilled student models against the C-RADIOv4-H teacher across representation alignment, ImageNet classification probes, COCO detection transfer, and efficiency profiling.

All scripts are runnable as `python -m bench.<module>` from the repo root.

---

## Prerequisites

```bash
# Core dependencies (same as distill/)
pip install torch torchvision timm transformers omegaconf

# COCO benchmarks
pip install pycocotools

# FLOPs counting (optional, for bench.eff.profile)
pip install fvcore
```

---

## Quick Start — One Command

Run the entire benchmark suite with a single command:

```bash
python -m bench.run_all \
    --student_ckpt path/to/best.pt \
    --imagenet_root path/to/imagenet \
    --coco_root path/to/coco \
    --out_dir path/to/bench_outputs
```

This runs all stages in order: alignment → efficiency → feature extraction → linear probe → kNN → detection train → detection eval → summary report. Each stage logs to `out_dir/logs/`.

**Skip or select stages:**

```bash
# Skip slow stages
python -m bench.run_all \
    --student_ckpt best.pt --imagenet_root /data/imagenet --coco_root /data/coco \
    --out_dir /tmp/bench --skip detection_train,detection_eval

# Run only alignment and efficiency
python -m bench.run_all \
    --student_ckpt best.pt --imagenet_root /data/imagenet --coco_root /data/coco \
    --out_dir /tmp/bench --only alignment,efficiency
```

**Available stages:** `alignment`, `efficiency`, `extract_train`, `extract_val`, `linear_probe`, `knn`, `detection_train`, `detection_eval`, `summary`

**Tunable defaults:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--feature` | `f3_pool` | Feature to extract (`f3_pool` or `summary`) |
| `--probe_epochs` | `50` | Linear probe training epochs |
| `--det_epochs` | `12` | Detection head training epochs |
| `--det_batch_size` | `8` | Detection training batch size |
| `--batch_size` | `32` | Default batch size for other stages |

---

## Individual Scripts

If you prefer running stages individually:

```bash
# Set common paths
export CKPT=path/to/student_checkpoint.pt
export IMAGENET=path/to/imagenet       # must have train/ and val/ subdirs
export COCO=path/to/coco               # must have train2017/, val2017/, annotations/
export OUT=path/to/bench_outputs
```

### 1. Representation Alignment

Computes 20+ metrics measuring how well the student matches the teacher's representations (cosine similarity, CKA, retrieval accuracy, high-frequency alignment, edge correlation, etc.).

```bash
# Evaluate on ImageNet validation set
python -m bench.rep.eval_alignment \
    --student_ckpt $CKPT \
    --dataset imagenet_val \
    --imagenet_root $IMAGENET \
    --batch_size 32 \
    --out_dir $OUT/alignment_imagenet

# Evaluate on COCO validation set
python -m bench.rep.eval_alignment \
    --student_ckpt $CKPT \
    --dataset coco_val \
    --coco_root $COCO \
    --out_dir $OUT/alignment_coco

# Evaluate on arbitrary image folder
python -m bench.rep.eval_alignment \
    --student_ckpt $CKPT \
    --dataset folder \
    --folder /path/to/images \
    --out_dir $OUT/alignment_custom
```

**Key output:** `metrics.json` with `alignment_score` (composite, same formula as training validation).

### 2. ImageNet Classification Probes

#### Step 1: Extract Features

```bash
# Student features (f3_pool = pooled stage-3, or summary = projected to teacher dim)
python -m bench.imagenet.extract_features \
    --model student \
    --student_ckpt $CKPT \
    --split train \
    --imagenet_root $IMAGENET \
    --feature f3_pool \
    --out_dir $OUT/feat_student_train

python -m bench.imagenet.extract_features \
    --model student \
    --student_ckpt $CKPT \
    --split val \
    --imagenet_root $IMAGENET \
    --feature f3_pool \
    --out_dir $OUT/feat_student_val

# Teacher features (always extracts summary)
python -m bench.imagenet.extract_features \
    --model teacher \
    --split train \
    --imagenet_root $IMAGENET \
    --out_dir $OUT/feat_teacher_train

python -m bench.imagenet.extract_features \
    --model teacher \
    --split val \
    --imagenet_root $IMAGENET \
    --out_dir $OUT/feat_teacher_val
```

Features are saved as sharded `.pt` files (one shard per ~50k samples).

#### Step 2a: Linear Probe

```bash
python -m bench.imagenet.linear_probe \
    --train_features_dir $OUT/feat_student_train \
    --val_features_dir $OUT/feat_student_val \
    --epochs 50 \
    --lr 0.1 \
    --batch_size 256 \
    --out_dir $OUT/linear_probe_student
```

**Key output:** `metrics.json` with `top1`, `top5`.

#### Step 2b: kNN Evaluation

```bash
python -m bench.imagenet.knn_eval \
    --train_features_dir $OUT/feat_student_train \
    --val_features_dir $OUT/feat_student_val \
    --k 20 \
    --out_dir $OUT/knn_student
```

**Key output:** `metrics.json` with `top1`, `top5`.

### 3. COCO Detection Transfer

#### Step 1: Train Detection Head

Trains FasterRCNN RPN + ROI heads on a frozen student backbone.

```bash
python -m bench.coco.det_train_headonly \
    --model student \
    --student_ckpt $CKPT \
    --coco_root $COCO \
    --epochs 12 \
    --batch_size 8 \
    --lr 0.02 \
    --out_dir $OUT/det_student
```

#### Step 2: Evaluate Detection

```bash
python -m bench.coco.det_eval \
    --det_ckpt $OUT/det_student/checkpoints/detector.pt \
    --student_ckpt $CKPT \
    --coco_root $COCO \
    --out_dir $OUT/det_eval_student
```

**Key output:** `metrics.json` with `mAP`, `AP50`, `AP75`, `APS`, `APM`, `APL`.

### 4. Efficiency Profiling

```bash
# Profile student only
python -m bench.eff.profile \
    --model student \
    --student_ckpt $CKPT \
    --out_dir $OUT/eff_student

# Profile both teacher and student (includes compression ratio + speedup)
python -m bench.eff.profile \
    --model both \
    --student_ckpt $CKPT \
    --out_dir $OUT/eff_both
```

**Key output:** `efficiency.json` with params, FLOPs, latency (ms), throughput (img/s), compression ratio, speedup.

### 5. Aggregate Results

```bash
python -m bench.reports.summarize_runs \
    --runs_root $OUT \
    --out_dir $OUT/report
```

**Key output:** `summary.csv` (flat table) + `summary.md` (paper-ready markdown tables).

---

## Recommended Workflow

### Full evaluation of a checkpoint

```bash
CKPT=path/to/best.pt
OUT=/tmp/bench_$(date +%Y%m%d)

# 1. Quick checks (minutes)
python -m bench.rep.eval_alignment --student_ckpt $CKPT --dataset imagenet_val --imagenet_root $IMAGENET --out_dir $OUT/align
python -m bench.eff.profile --model both --student_ckpt $CKPT --out_dir $OUT/eff

# 2. Classification probes (hours for train extraction + linear probe)
for SPLIT in train val; do
    python -m bench.imagenet.extract_features --model student --student_ckpt $CKPT --split $SPLIT --imagenet_root $IMAGENET --feature f3_pool --out_dir $OUT/feat_$SPLIT
done
python -m bench.imagenet.linear_probe --train_features_dir $OUT/feat_train --val_features_dir $OUT/feat_val --out_dir $OUT/linear
python -m bench.imagenet.knn_eval --train_features_dir $OUT/feat_train --val_features_dir $OUT/feat_val --out_dir $OUT/knn

# 3. Detection transfer (hours)
python -m bench.coco.det_train_headonly --model student --student_ckpt $CKPT --coco_root $COCO --out_dir $OUT/det
python -m bench.coco.det_eval --det_ckpt $OUT/det/checkpoints/detector.pt --student_ckpt $CKPT --coco_root $COCO --out_dir $OUT/det_eval

# 4. Summarize everything
python -m bench.reports.summarize_runs --runs_root $OUT --out_dir $OUT/report
cat $OUT/report/summary.md
```

### Comparing two checkpoints

Run the full workflow above for each checkpoint into separate `$OUT` dirs, then point `summarize_runs` at the parent:

```bash
python -m bench.reports.summarize_runs --runs_root /path/to/all_runs --out_dir /path/to/comparison
```

---

## Output Convention

Every script produces outputs in `--out_dir` following this structure:

```
out_dir/
  run_meta.json    — timestamp, git hash, CLI args
  metrics.json     — aggregated results (or efficiency.json for profiling)
  metrics.jsonl    — per-batch metrics (where applicable)
  checkpoints/     — saved model weights (probe/detector heads)
```

---

## CLI Reference

All scripts accept `--help` for full argument documentation:

```bash
python -m bench.rep.eval_alignment --help
python -m bench.imagenet.extract_features --help
python -m bench.imagenet.linear_probe --help
python -m bench.imagenet.knn_eval --help
python -m bench.coco.det_train_headonly --help
python -m bench.coco.det_eval --help
python -m bench.eff.profile --help
python -m bench.reports.summarize_runs --help
```

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher_id` | `nvidia/C-RADIOv4-H` | HuggingFace teacher model ID |
| `--student_ckpt` | (required) | Path to student `.pt` checkpoint |
| `--size` | `416` | Input image size |
| `--patch_size` | `16` | Patch size (determines spatial grid: size/patch_size) |
| `--batch_size` | `32` | Batch size |
| `--num_workers` | `8` | DataLoader workers |
| `--amp` / `--no_amp` | `--amp` | Mixed precision |
| `--out_dir` | (required) | Output directory |

---

## Architecture

```
bench/
├── common/              Shared utilities (no scripts)
│   ├── config.py        Constants, argparse helpers
│   ├── model_loaders.py Load teacher/student models
│   ├── preprocess.py    Data loading (re-exports from distill.data)
│   ├── io.py            JSON/JSONL, run metadata
│   ├── metrics.py       Metric functions (re-exports from distill.model)
│   └── timing.py        Latency/throughput measurement
├── rep/
│   └── eval_alignment.py   Representation alignment eval
├── imagenet/
│   ├── extract_features.py Feature extraction to disk
│   ├── linear_probe.py     Linear probe training
│   └── knn_eval.py         Cosine kNN evaluation
├── coco/
│   ├── coco_index.py       Dataset + collate + category mapping
│   ├── det_train_headonly.py FasterRCNN head-only training
│   └── det_eval.py         COCO mAP evaluation
├── eff/
│   └── profile.py          Params/FLOPs/latency/throughput
└── reports/
    └── summarize_runs.py   Aggregate → summary.csv + summary.md
```

All metric and data functions are imported from `distill/` — no copy-paste.
