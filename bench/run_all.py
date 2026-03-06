"""
Run the full benchmark suite end-to-end.

Usage:
    python -m bench.run_all \
        --student_ckpt path/to/best.pt \
        --imagenet_root path/to/imagenet \
        --coco_root path/to/coco \
        --out_dir path/to/bench_outputs

Optional:
    --skip alignment,detection   Skip specific stages
    --size 416                   Input image size
    --feature f3_pool            Feature type for ImageNet probes
    --det_epochs 12              Detection head training epochs
    --probe_epochs 50            Linear probe epochs
    --batch_size 32              Default batch size (detection uses 8)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from bench.common.io import save_json, get_git_hash


STAGES = [
    "alignment",
    "efficiency",
    "extract_train",
    "extract_val",
    "linear_probe",
    "knn",
    "detection_train",
    "detection_eval",
    "summary",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run full benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--student_ckpt", type=str, required=True,
                        help="Path to student checkpoint (.pt)")
    parser.add_argument("--imagenet_root", type=str, required=True,
                        help="ImageNet root (must contain train/ and val/)")
    parser.add_argument("--coco_root", type=str, required=True,
                        help="COCO root (must contain train2017/, val2017/, annotations/)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for all benchmark results")

    parser.add_argument("--teacher_id", type=str, default="nvidia/C-RADIOv4-H")
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--feature", type=str, default="f3_pool",
                        choices=["f3_pool", "summary"])
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--det_epochs", type=int, default=12)
    parser.add_argument("--det_batch_size", type=int, default=8)
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated stages to skip: " + ",".join(STAGES))
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated stages to run (overrides --skip)")
    return parser


def run_step(name: str, cmd: list[str], log_dir: Path) -> bool:
    """Run a subprocess, stream output, log to file. Returns True on success."""
    log_file = log_dir / f"{name}.log"
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"  CMD:   {' '.join(cmd)}")
    print(f"  LOG:   {log_file}")
    print(f"{'='*60}\n")

    t0 = time.time()
    with open(log_file, "w") as f:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        f.write(proc.stdout)
        # Also print to console
        print(proc.stdout)

    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {proc.returncode}, {elapsed:.0f}s)")
        print(f"  See log: {log_file}")
        return False

    print(f"\n  DONE: {name} ({elapsed:.0f}s)")
    return True


def main():
    parser = build_parser()
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(exist_ok=True)

    py = sys.executable

    # Determine which stages to run
    if args.only:
        active = set(args.only.split(","))
    elif args.skip:
        active = set(STAGES) - set(args.skip.split(","))
    else:
        active = set(STAGES)

    # Save top-level run metadata
    save_json({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_hash": get_git_hash(),
        "args": vars(args),
        "stages": sorted(active),
    }, str(out / "run_meta.json"))

    results = {}
    t_total = time.time()

    # ── 1. Representation Alignment ──────────────────────────────
    if "alignment" in active:
        ok = run_step("alignment", [
            py, "-m", "bench.rep.eval_alignment",
            "--student_ckpt", args.student_ckpt,
            "--teacher_id", args.teacher_id,
            "--dataset", "imagenet_val",
            "--imagenet_root", args.imagenet_root,
            "--size", str(args.size),
            "--patch_size", str(args.patch_size),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--out_dir", str(out / "alignment"),
        ], log_dir)
        results["alignment"] = "ok" if ok else "FAILED"

    # ── 2. Efficiency Profiling ──────────────────────────────────
    if "efficiency" in active:
        ok = run_step("efficiency", [
            py, "-m", "bench.eff.profile",
            "--model", "both",
            "--student_ckpt", args.student_ckpt,
            "--teacher_id", args.teacher_id,
            "--size", str(args.size),
            "--patch_size", str(args.patch_size),
            "--out_dir", str(out / "efficiency"),
        ], log_dir)
        results["efficiency"] = "ok" if ok else "FAILED"

    # ── 3. Extract Features (train + val) ────────────────────────
    feat_train = str(out / "feat_train")
    feat_val = str(out / "feat_val")

    if "extract_train" in active:
        ok = run_step("extract_train", [
            py, "-m", "bench.imagenet.extract_features",
            "--model", "student",
            "--student_ckpt", args.student_ckpt,
            "--split", "train",
            "--imagenet_root", args.imagenet_root,
            "--feature", args.feature,
            "--size", str(args.size),
            "--patch_size", str(args.patch_size),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--out_dir", feat_train,
        ], log_dir)
        results["extract_train"] = "ok" if ok else "FAILED"

    if "extract_val" in active:
        ok = run_step("extract_val", [
            py, "-m", "bench.imagenet.extract_features",
            "--model", "student",
            "--student_ckpt", args.student_ckpt,
            "--split", "val",
            "--imagenet_root", args.imagenet_root,
            "--feature", args.feature,
            "--size", str(args.size),
            "--patch_size", str(args.patch_size),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--out_dir", feat_val,
        ], log_dir)
        results["extract_val"] = "ok" if ok else "FAILED"

    # ── 4. Linear Probe ──────────────────────────────────────────
    if "linear_probe" in active:
        ok = run_step("linear_probe", [
            py, "-m", "bench.imagenet.linear_probe",
            "--train_features_dir", feat_train,
            "--val_features_dir", feat_val,
            "--epochs", str(args.probe_epochs),
            "--batch_size", "256",
            "--out_dir", str(out / "linear_probe"),
        ], log_dir)
        results["linear_probe"] = "ok" if ok else "FAILED"

    # ── 5. kNN Evaluation ────────────────────────────────────────
    if "knn" in active:
        ok = run_step("knn", [
            py, "-m", "bench.imagenet.knn_eval",
            "--train_features_dir", feat_train,
            "--val_features_dir", feat_val,
            "--k", "20",
            "--out_dir", str(out / "knn"),
        ], log_dir)
        results["knn"] = "ok" if ok else "FAILED"

    # ── 6. COCO Detection Training ───────────────────────────────
    det_ckpt = str(out / "detection" / "checkpoints" / "detector.pt")

    if "detection_train" in active:
        ok = run_step("detection_train", [
            py, "-m", "bench.coco.det_train_headonly",
            "--model", "student",
            "--student_ckpt", args.student_ckpt,
            "--coco_root", args.coco_root,
            "--size", str(args.size),
            "--patch_size", str(args.patch_size),
            "--epochs", str(args.det_epochs),
            "--batch_size", str(args.det_batch_size),
            "--out_dir", str(out / "detection"),
        ], log_dir)
        results["detection_train"] = "ok" if ok else "FAILED"

    # ── 7. COCO Detection Evaluation ─────────────────────────────
    if "detection_eval" in active:
        ok = run_step("detection_eval", [
            py, "-m", "bench.coco.det_eval",
            "--det_ckpt", det_ckpt,
            "--student_ckpt", args.student_ckpt,
            "--coco_root", args.coco_root,
            "--size", str(args.size),
            "--patch_size", str(args.patch_size),
            "--out_dir", str(out / "detection_eval"),
        ], log_dir)
        results["detection_eval"] = "ok" if ok else "FAILED"

    # ── 8. Summary Report ────────────────────────────────────────
    if "summary" in active:
        ok = run_step("summary", [
            py, "-m", "bench.reports.summarize_runs",
            "--runs_root", str(out),
            "--out_dir", str(out / "report"),
        ], log_dir)
        results["summary"] = "ok" if ok else "FAILED"

    # ── Final Summary ────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ALL STAGES COMPLETE  ({elapsed_total/60:.1f} min)")
    print(f"{'='*60}")
    for stage, status in results.items():
        marker = "+" if status == "ok" else "X"
        print(f"  [{marker}] {stage}")
    print(f"\n  Output: {out}")

    report_md = out / "report" / "summary.md"
    if report_md.exists():
        print(f"  Report: {report_md}")

    save_json(results, str(out / "stage_results.json"))

    if any(v == "FAILED" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
