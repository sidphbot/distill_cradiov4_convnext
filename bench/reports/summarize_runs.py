"""
Aggregate benchmark metrics into summary tables.

Crawls run directories for metrics.json / efficiency.json and produces
summary.csv + summary.md (paper-ready markdown tables).

Usage:
    python -m bench.reports.summarize_runs \
        --runs_root /tmp/bench_runs --out_dir /tmp/report
"""

import argparse
import csv
from pathlib import Path

from bench.common.io import load_json, save_json


BENCHMARK_TYPES = {
    "alignment": "metrics.json",
    "linear_probe": "metrics.json",
    "knn": "metrics.json",
    "detection": "metrics.json",
    "efficiency": "efficiency.json",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize benchmark runs")
    parser.add_argument("--runs_root", type=str, required=True,
                        help="Root directory containing benchmark run subdirs")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser


def detect_benchmark_type(run_dir: Path) -> str:
    """Infer benchmark type from run directory contents and run_meta."""
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = load_json(str(meta_path))
        args = meta.get("args", {})
        if isinstance(args, dict):
            # Detect by CLI args
            if "alignment_score" in str(load_json(str(run_dir / "metrics.json")) if (run_dir / "metrics.json").exists() else {}):
                return "alignment"
            if "train_features_dir" in args and "k" in args:
                return "knn"
            if "train_features_dir" in args:
                return "linear_probe"
            if "det_ckpt" in args or "coco_root" in args:
                return "detection"

    if (run_dir / "efficiency.json").exists():
        return "efficiency"
    if (run_dir / "metrics.json").exists():
        metrics = load_json(str(run_dir / "metrics.json"))
        if "alignment_score" in metrics:
            return "alignment"
        if "mAP" in metrics:
            return "detection"
        if "top1" in metrics:
            if "k" in metrics:
                return "knn"
            return "linear_probe"
    return "unknown"


def collect_runs(runs_root: Path) -> dict:
    """Collect all runs grouped by benchmark type."""
    groups = {}
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        btype = detect_benchmark_type(run_dir)
        if btype == "unknown":
            continue
        if btype not in groups:
            groups[btype] = []

        entry = {"run": run_dir.name, "dir": str(run_dir)}

        metrics_file = "efficiency.json" if btype == "efficiency" else "metrics.json"
        metrics_path = run_dir / metrics_file
        if metrics_path.exists():
            entry["metrics"] = load_json(str(metrics_path))

        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            entry["meta"] = load_json(str(meta_path))

        groups[btype].append(entry)
    return groups


def write_csv(groups: dict, out_path: str):
    """Write flat summary CSV."""
    rows = []
    for btype, entries in groups.items():
        for entry in entries:
            row = {"benchmark": btype, "run": entry["run"]}
            metrics = entry.get("metrics", {})
            # Flatten nested dicts (e.g. efficiency has teacher/student sub-dicts)
            for k, v in metrics.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        row[f"{k}_{k2}"] = v2
                else:
                    row[k] = v
            rows.append(row)

    if not rows:
        return

    all_keys = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(groups: dict, out_path: str):
    """Write paper-ready markdown summary."""
    lines = ["# Benchmark Summary\n"]

    if "alignment" in groups:
        lines.append("## Representation Alignment\n")
        lines.append("| Run | Alignment Score | Summary Cos | Spatial Cos | HF Cos | Act F1 |")
        lines.append("|-----|----------------|-------------|-------------|--------|--------|")
        for e in groups["alignment"]:
            m = e.get("metrics", {})
            lines.append(
                f"| {e['run']} "
                f"| {m.get('alignment_score', 'N/A'):.4f} "
                f"| {m.get('summary_cos_mean', 'N/A'):.4f} "
                f"| {m.get('spatial_cos_mean', 'N/A'):.4f} "
                f"| {m.get('hf_cos', 'N/A'):.4f} "
                f"| {m.get('act_f1', 'N/A'):.4f} |"
            )
        lines.append("")

    if "linear_probe" in groups:
        lines.append("## ImageNet Linear Probe\n")
        lines.append("| Run | Top-1 | Top-5 | Epochs |")
        lines.append("|-----|-------|-------|--------|")
        for e in groups["linear_probe"]:
            m = e.get("metrics", {})
            lines.append(
                f"| {e['run']} "
                f"| {m.get('top1', 'N/A'):.4f} "
                f"| {m.get('top5', 'N/A'):.4f} "
                f"| {m.get('epochs', 'N/A')} |"
            )
        lines.append("")

    if "knn" in groups:
        lines.append("## ImageNet kNN\n")
        lines.append("| Run | Top-1 | Top-5 | k |")
        lines.append("|-----|-------|-------|---|")
        for e in groups["knn"]:
            m = e.get("metrics", {})
            lines.append(
                f"| {e['run']} "
                f"| {m.get('top1', 'N/A'):.4f} "
                f"| {m.get('top5', 'N/A'):.4f} "
                f"| {m.get('k', 'N/A')} |"
            )
        lines.append("")

    if "detection" in groups:
        lines.append("## COCO Detection\n")
        lines.append("| Run | mAP | AP50 | AP75 | APS | APM | APL |")
        lines.append("|-----|-----|------|------|-----|-----|-----|")
        for e in groups["detection"]:
            m = e.get("metrics", {})
            lines.append(
                f"| {e['run']} "
                f"| {m.get('mAP', 'N/A'):.4f} "
                f"| {m.get('AP50', 'N/A'):.4f} "
                f"| {m.get('AP75', 'N/A'):.4f} "
                f"| {m.get('APS', 'N/A'):.4f} "
                f"| {m.get('APM', 'N/A'):.4f} "
                f"| {m.get('APL', 'N/A'):.4f} |"
            )
        lines.append("")

    if "efficiency" in groups:
        lines.append("## Efficiency\n")
        lines.append("| Run | Model | Params | FLOPs | Latency (ms) | Throughput (img/s) |")
        lines.append("|-----|-------|--------|-------|-------------|-------------------|")
        for e in groups["efficiency"]:
            m = e.get("metrics", {})
            for model_key in ("teacher", "student"):
                if model_key in m:
                    info = m[model_key]
                    lines.append(
                        f"| {e['run']} "
                        f"| {model_key} "
                        f"| {info.get('total_params', 'N/A'):,} "
                        f"| {info.get('flops', 'N/A')} "
                        f"| {info.get('latency_ms_mean', 'N/A'):.2f} "
                        f"| {info.get('throughput_img_per_sec', 'N/A'):.1f} |"
                    )
            if "compression_ratio" in m:
                lines.append(f"\nCompression ratio: {m['compression_ratio']:.1f}x, Speedup: {m.get('speedup', 'N/A'):.1f}x\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def summarize(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_root = Path(args.runs_root)
    groups = collect_runs(runs_root)

    if not groups:
        print(f"No benchmark runs found in {runs_root}")
        return

    csv_path = str(out_dir / "summary.csv")
    md_path = str(out_dir / "summary.md")

    write_csv(groups, csv_path)
    write_markdown(groups, md_path)

    print(f"Found {sum(len(v) for v in groups.values())} runs across {len(groups)} benchmark types")
    for btype, entries in groups.items():
        print(f"  {btype}: {len(entries)} runs")
    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
