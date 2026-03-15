"""
Standalone representation alignment evaluation.

Mirrors lightning_module.py validation_step as a standalone script.
Computes all alignment metrics between teacher and student on a given dataset.

Usage:
    python -m bench.rep.eval_alignment \
        --student_ckpt checkpoints/best.pt \
        --dataset imagenet_val --imagenet_root /data/imagenet \
        --out_dir /tmp/align_eval
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from distill.model import teacher_forward_fixed
from bench.common.config import add_teacher_args, add_student_args, add_data_args, add_output_args, parse_amp
from bench.common.model_loaders import load_teacher, load_student
from bench.common.preprocess import collect_images, normalize_batch, make_eval_dataloader
from bench.common.io import save_json, append_jsonl, save_run_meta
from bench.common.metrics import (
    flatten_spatial_tokens, cosine_sim, batch_retrieval_top1, batch_retrieval_mrr,
    centered_kernel_alignment, style_gram_loss, spatial_energy,
    corrcoef_1d, topk_activation_f1, pearson_corr_per_sample,
    sobel_mag_2d, laplacian_highpass_depthwise,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Representation alignment evaluation")
    add_teacher_args(parser)
    add_student_args(parser)
    add_data_args(parser)
    add_output_args(parser)
    parser.add_argument("--dataset", type=str, default="imagenet_val",
                        choices=["imagenet_val", "coco_val", "folder"])
    parser.add_argument("--imagenet_root", type=str, default="")
    parser.add_argument("--coco_root", type=str, default="")
    parser.add_argument("--folder", type=str, default="")
    return parser


def resolve_image_paths(args) -> list:
    if args.dataset == "imagenet_val":
        root = Path(args.imagenet_root) / "val"
        return collect_images(str(root))
    elif args.dataset == "coco_val":
        root = Path(args.coco_root) / "val2017"
        return collect_images(str(root))
    elif args.dataset == "folder":
        return collect_images(args.folder)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


@torch.no_grad()
def evaluate(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_meta(str(out_dir), args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = parse_amp(args) and device == "cuda"

    # Load models
    print("Loading teacher...")
    teacher_b = load_teacher(args.teacher_id, device, use_amp)
    print("Loading student...")
    student_b = load_student(args.student_ckpt, device, args.size, args.patch_size)
    Ht, Wt = student_b.Ht, student_b.Wt

    # Build dataloader
    paths = resolve_image_paths(args)
    print(f"Found {len(paths)} images")
    loader = make_eval_dataloader(paths, args.size, args.batch_size, args.num_workers)

    # Accumulators
    n = 0
    scalar_sums = {}
    summary_cos_all = []
    spatial_cos_all = []
    edge_corr_all = []
    loss_sums = {}

    jsonl_path = str(out_dir / "metrics.jsonl")

    for batch_idx, (img_u8, pil_rs, keys) in enumerate(tqdm(loader, desc="Evaluating")):
        B = img_u8.shape[0]

        # Teacher forward
        t_summary, t_spatial_tokens = teacher_forward_fixed(
            teacher_b.model, teacher_b.processor, pil_rs,
            device, args.size, use_amp,
        )
        del pil_rs

        # Student preprocessing
        x = img_u8.float().div_(255.0)
        del img_u8
        normalize_batch(x)
        x = x.to(device, non_blocking=True)

        t_summary = t_summary.to(device, non_blocking=True).float()
        t_spatial_tokens = t_spatial_tokens.to(device, non_blocking=True).float()

        # Student forward
        with torch.amp.autocast("cuda", enabled=use_amp):
            s_sum, s_tokens, s_sp, (f2, f3) = student_b.model(x)

        # Move to CPU for metrics
        s_sum_c = s_sum.float().cpu()
        t_summary_c = t_summary.float().cpu()
        s_sp_c = s_sp.float().cpu()
        t_spatial_c = t_spatial_tokens.float().cpu()
        x_c = x.float().cpu()
        del s_sum, t_summary, s_tokens, s_sp, t_spatial_tokens, x, f2, f3
        torch.cuda.empty_cache()

        n += 1

        # Spatial tokens
        s_tokens2 = flatten_spatial_tokens(s_sp_c)
        t_tokens2 = t_spatial_c
        Bt, Tt, Dt_ = t_tokens2.shape
        t_map = t_tokens2.transpose(1, 2).contiguous().reshape(Bt, Dt_, Ht, Wt)

        # Summary metrics
        batch_summary_cos = cosine_sim(s_sum_c, t_summary_c, dim=-1)
        summary_cos_all.append(batch_summary_cos)

        batch_spatial_cos = cosine_sim(s_tokens2, t_tokens2, dim=-1).flatten()
        spatial_cos_all.append(batch_spatial_cos)

        def accum(key, val):
            scalar_sums[key] = scalar_sums.get(key, 0.0) + float(val)

        accum("retrieval_top1", batch_retrieval_top1(s_sum_c, t_summary_c))
        accum("retrieval_mrr", batch_retrieval_mrr(s_sum_c, t_summary_c))
        accum("summary_cka", centered_kernel_alignment(s_sum_c, t_summary_c))
        accum("summary_mse_mean", ((s_sum_c - t_summary_c) ** 2).mean())
        accum("summary_norm_ratio", (s_sum_c.norm(dim=-1) / (t_summary_c.norm(dim=-1) + 1e-8)).mean())
        accum("summary_mean_abs_diff", (s_sum_c.mean(dim=0) - t_summary_c.mean(dim=0)).abs().mean())
        accum("summary_std_abs_diff", (s_sum_c.std(dim=0) - t_summary_c.std(dim=0)).abs().mean())

        # Spatial metrics
        accum("spatial_mse_mean", ((s_tokens2 - t_tokens2) ** 2).mean())
        accum("spatial_norm_ratio", (s_tokens2.norm(dim=-1) / (t_tokens2.norm(dim=-1) + 1e-8)).mean())
        accum("spatial_style_gram", style_gram_loss(s_tokens2, t_tokens2))

        s_en = spatial_energy(s_sp_c)
        t_en = spatial_energy(t_map)
        accum("spatial_energy_ratio", (s_en / (t_en + 1e-8)).mean())

        s_meanD = s_tokens2.mean(dim=1)
        t_meanD = t_tokens2.mean(dim=1)
        accum("spatial_meanD_corr", corrcoef_1d(s_meanD.flatten(), t_meanD.flatten()))
        accum("act_f1", topk_activation_f1(s_tokens2, t_tokens2))

        del s_sum_c, t_summary_c, s_tokens2, t_tokens2, s_meanD, t_meanD

        # Edge alignment
        x_gray = x_c.mean(dim=1, keepdim=True)
        g_img = sobel_mag_2d(x_gray)
        del x_c, x_gray
        s_energy_map = torch.sqrt((s_sp_c * s_sp_c).sum(dim=1, keepdim=True) + 1e-8)
        g_feat = sobel_mag_2d(s_energy_map)
        del s_energy_map
        g_img_ds = F.interpolate(g_img, size=(Ht, Wt), mode="bilinear", align_corners=False)
        del g_img
        edge_corr_all.append(pearson_corr_per_sample(g_feat.flatten(1), g_img_ds.flatten(1)))
        del g_feat, g_img_ds

        # High-frequency metrics
        s_hf = laplacian_highpass_depthwise(s_sp_c)
        t_hf = laplacian_highpass_depthwise(t_map)
        del s_sp_c, t_map
        s_hf_tok = flatten_spatial_tokens(s_hf)
        t_hf_tok = flatten_spatial_tokens(t_hf)
        accum("hf_cos", cosine_sim(s_hf_tok, t_hf_tok, dim=-1).mean())
        accum("hf_mse", F.mse_loss(s_hf, t_hf))
        s_hf_en = spatial_energy(s_hf)
        t_hf_en = spatial_energy(t_hf)
        accum("hf_energy_ratio", (s_hf_en / (t_hf_en + 1e-8)).mean())
        del s_hf, t_hf, s_hf_tok, t_hf_tok

        # Per-batch JSONL
        batch_metrics = {k: v / n for k, v in scalar_sums.items()}
        batch_metrics["batch_idx"] = batch_idx
        append_jsonl(batch_metrics, jsonl_path)

    # Aggregate
    results = {k: v / n for k, v in scalar_sums.items()}

    # Cosine stats
    all_sum_cos = torch.cat(summary_cos_all)
    all_sp_cos = torch.cat(spatial_cos_all)
    all_edge = torch.cat(edge_corr_all)

    def safe_quantile(t, q):
        """Quantile that handles tensors too large for torch.quantile."""
        if t.numel() <= 2**24:
            return float(t.quantile(q))
        # Subsample for large tensors
        idx = torch.randperm(t.numel())[:2**24]
        return float(t.flatten()[idx].quantile(q))

    results["summary_cos_mean"] = float(all_sum_cos.mean())
    results["summary_cos_q05"] = safe_quantile(all_sum_cos, 0.05)
    results["summary_cos_q95"] = safe_quantile(all_sum_cos, 0.95)
    results["spatial_cos_mean"] = float(all_sp_cos.mean())
    results["spatial_cos_q05"] = safe_quantile(all_sp_cos, 0.05)
    results["spatial_cos_q95"] = safe_quantile(all_sp_cos, 0.95)
    results["edge_corr_mean"] = float(all_edge.mean())

    # Composite alignment score (same weights as lightning_module)
    cos_sum_sim = results["summary_cos_mean"]
    cos_sp_sim = results["spatial_cos_mean"]
    hf_cos_mean = results["hf_cos"]
    hf_mse_mean = results["hf_mse"]
    act_f1_mean = results["act_f1"]

    alignment_score = (
        0.30 * cos_sum_sim + 0.30 * cos_sp_sim
        + 0.20 * hf_cos_mean - 0.10 * hf_mse_mean
        + 0.30 * act_f1_mean
    )
    results["alignment_score"] = alignment_score
    results["n_batches"] = n

    save_json(results, str(out_dir / "metrics.json"))
    print(f"\nAlignment score: {alignment_score:.4f}")
    print(f"Results saved to {out_dir / 'metrics.json'}")
    return results


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
