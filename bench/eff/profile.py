"""
Efficiency profiling: params, FLOPs, latency, throughput.

Usage:
    python -m bench.eff.profile \
        --model student --student_ckpt checkpoints/best.pt \
        --out_dir /tmp/eff
"""

import argparse
from pathlib import Path

import torch

from bench.common.config import (
    add_teacher_args, add_student_args, add_output_args,
    DEFAULT_SIZE, parse_amp,
)
from bench.common.model_loaders import load_teacher, load_student
from bench.common.timing import measure_latency, measure_throughput
from bench.common.io import save_json, save_run_meta


def build_parser():
    parser = argparse.ArgumentParser(description="Model efficiency profiling")
    parser.add_argument("--model", type=str, required=True,
                        choices=["teacher", "student", "both"])
    add_teacher_args(parser)
    parser.add_argument("--student_ckpt", type=str, default="")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--patch_size", type=int, default=16)
    add_output_args(parser)
    return parser


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def count_flops(model, input_shape, device):
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(1, *input_shape, device=device)
        flops = FlopCountAnalysis(model, x)
        return {"flops": flops.total()}
    except ImportError:
        print("fvcore not installed, skipping FLOPs count")
        return {"flops": "N/A (install fvcore)"}
    except Exception as e:
        print(f"FLOPs count failed: {e}")
        return {"flops": f"error: {e}"}


def profile_model(name, model, input_shape, device):
    """Profile a single model."""
    print(f"\nProfiling {name}...")
    result = {"name": name}

    # Params
    params = count_params(model)
    result.update(params)
    print(f"  Params: {params['total_params']:,} total, {params['trainable_params']:,} trainable")

    # FLOPs
    flops = count_flops(model, input_shape, device)
    result.update(flops)
    if isinstance(flops["flops"], (int, float)):
        print(f"  FLOPs: {flops['flops']:,.0f}")

    # Latency (batch_size=1)
    x1 = torch.randn(1, *input_shape, device=device)
    lat = measure_latency(model, x1)
    result.update(lat)
    print(f"  Latency: {lat['latency_ms_mean']:.2f} ± {lat['latency_ms_std']:.2f} ms")

    # Throughput (batch_size=32)
    thr = measure_throughput(model, input_shape, batch_size=32, device=device)
    result.update(thr)
    print(f"  Throughput: {thr['throughput_img_per_sec']:.1f} img/s")

    return result


@torch.no_grad()
def profile(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_meta(str(out_dir), args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_shape = (3, args.size, args.size)
    results = {}

    if args.model in ("teacher", "both"):
        teacher_b = load_teacher(args.teacher_id, device, amp=False)
        # Teacher expects processed inputs, profile with float32
        teacher_result = profile_model("teacher", teacher_b.model, input_shape, device)
        results["teacher"] = teacher_result

    if args.model in ("student", "both"):
        student_b = load_student(args.student_ckpt, device, args.size, args.patch_size)
        student_result = profile_model("student", student_b.model, input_shape, device)
        results["student"] = student_result

    if "teacher" in results and "student" in results:
        t_params = results["teacher"]["total_params"]
        s_params = results["student"]["total_params"]
        results["compression_ratio"] = t_params / max(s_params, 1)
        t_lat = results["teacher"]["latency_ms_mean"]
        s_lat = results["student"]["latency_ms_mean"]
        results["speedup"] = t_lat / max(s_lat, 1e-6)
        print(f"\nCompression: {results['compression_ratio']:.1f}x params")
        print(f"Speedup: {results['speedup']:.1f}x latency")

    save_json(results, str(out_dir / "efficiency.json"))
    print(f"\nResults saved to {out_dir / 'efficiency.json'}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    profile(args)


if __name__ == "__main__":
    main()
