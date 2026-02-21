#!/usr/bin/env python3
# distill_convnext_from_cache.py
import os
import io
import glob
import time
import math
import random
import tarfile
import argparse
from pathlib import Path
from typing import List, Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import IterableDataset, DataLoader

from PIL import Image

import timm
from timm.data import resolve_data_config

import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
import torch.nn.functional as F

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

def preprocess_uint8(img_u8: torch.Tensor, size=416, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    # img_u8: (C,H,W) uint8
    x = img_u8.float().div(255.0)
    x = TF.resize(x, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    x = TF.normalize(x, mean=mean, std=std)
    return x


def build_param_groups(model: torch.nn.Module, weight_decay: float):
    """
    Exclude bias and norm parameters from weight decay.
    Common rule: no decay for 1D params (norm weights) and explicit biases.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def _flatten_spatial_tokens(x_bdhw: torch.Tensor) -> torch.Tensor:
    # (B,D,H,W) -> (B,T,D)
    B, D, H, W = x_bdhw.shape
    return x_bdhw.permute(0, 2, 3, 1).reshape(B, H*W, D)

def _cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    a = a / (a.norm(dim=dim, keepdim=True) + eps)
    b = b / (b.norm(dim=dim, keepdim=True) + eps)
    return (a * b).sum(dim=dim)

def _corrcoef_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x,y: (N,)
    x = x - x.mean()
    y = y - y.mean()
    return (x @ y) / ((x.norm() + eps) * (y.norm() + eps))

def _batch_retrieval_top1(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """
    z_s, z_t: (B, D) embeddings (assumed already comparable dims)
    Computes top-1 agreement: for each i, argmax_j cos(z_s[i], z_t[j]) == i
    """
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    sim = z_s @ z_t.T  # (B,B)
    nn = sim.argmax(dim=1)
    gt = torch.arange(z_s.shape[0], device=z_s.device)
    return (nn == gt).float().mean()

def _batch_retrieval_mrr(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """
    Mean Reciprocal Rank using teacher as targets.
    """
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    sim = z_s @ z_t.T  # (B,B)
    # ranks of diagonal element in each row
    # rank = 1 + number of elements strictly greater than sim[i,i]
    diag = sim.diag().unsqueeze(1)
    rank = 1 + (sim > diag).sum(dim=1)
    return (1.0 / rank.float()).mean()

def _gram_matrix(x_btd: torch.Tensor) -> torch.Tensor:
    """
    x_btd: (B,T,D). Returns per-sample Gram (B, D, D) normalized by T.
    """
    B, T, D = x_btd.shape
    x = x_btd
    G = torch.bmm(x.transpose(1,2), x) / float(T)  # (B,D,D)
    return G

def _style_gram_loss(s_btd: torch.Tensor, t_btd: torch.Tensor) -> torch.Tensor:
    Gs = _gram_matrix(s_btd)
    Gt = _gram_matrix(t_btd)
    return F.mse_loss(Gs, Gt)

def _centered_kernel_alignment(z_s: torch.Tensor, z_t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    CKA (linear) between student/teacher embeddings in batch.
    z_s,z_t: (B,D)
    """
    X = z_s - z_s.mean(dim=0, keepdim=True)
    Y = z_t - z_t.mean(dim=0, keepdim=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic = (K * L).sum()
    norm = torch.sqrt((K * K).sum() * (L * L).sum() + eps)
    return hsic / norm

def _spatial_energy(x: torch.Tensor) -> torch.Tensor:
    """
    Simple edge/texture energy proxy: mean gradient magnitude.
    x: (B,D,H,W) -> returns (B,) energy
    """
    dx = x[..., 1:] - x[..., :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    # pad to same size
    dx = F.pad(dx, (0,1,0,0))
    dy = F.pad(dy, (0,0,0,1))
    g = torch.sqrt(dx*dx + dy*dy + 1e-8)
    return g.mean(dim=(1,2,3))

def _to_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: (H,W)
    x = x - x.min()
    x = x / (x.max() + eps)
    return x

@torch.no_grad()
def log_spatial_compare_first_sample(
    tb_writer,
    step: int,
    s_spatial_bdhw: torch.Tensor,      # (B, Dt, Ht, Wt) student spatial (already aligned to teacher grid)
    t_tokens_btd: torch.Tensor,        # (B, T, Dt) teacher tokens
    Ht: int,
    Wt: int,
    tag_prefix: str = "val/spatial_compare",
    max_side: int = 512,
):
    """
    Logs:
      - cosine similarity heatmap (1 x H x W)
      - L2 error heatmap (1 x H x W)
      - optional RGB composite (3 x H x W): [cos, l2, 0]
      - scalars for the first sample
    """
    assert s_spatial_bdhw.ndim == 4
    assert t_tokens_btd.ndim == 3

    # First sample
    s = s_spatial_bdhw[0]          # (Dt, Ht, Wt)
    t_tokens = t_tokens_btd[0]     # (T, Dt)
    assert t_tokens.shape[0] == Ht * Wt, f"T={t_tokens.shape[0]} != Ht*Wt={Ht*Wt}"

    # Teacher: (T,Dt) -> (Dt,Ht,Wt)
    t = t_tokens.transpose(0, 1).contiguous().reshape(-1, Ht, Wt)  # (Dt,Ht,Wt)

    # Per-pixel cosine similarity over channel dimension
    # cos_map: (Ht,Wt)
    s_n = F.normalize(s, dim=0)  # normalize across channels
    t_n = F.normalize(t, dim=0)
    cos_map = (s_n * t_n).sum(dim=0)              # [-1,1]
    cos_map01 = (cos_map + 1.0) * 0.5             # [0,1]

    # Per-pixel L2 error over channel dimension
    l2_map = torch.sqrt(((s - t) ** 2).sum(dim=0) + 1e-8)          # >=0
    l2_map01 = _to_01(l2_map)

    # Make 1-channel images (C,H,W)
    cos_img = cos_map01.unsqueeze(0).clamp(0, 1)   # (1,H,W)
    l2_img = l2_map01.unsqueeze(0).clamp(0, 1)     # (1,H,W)

    # Upsample to max 512 on longer side (keeping aspect)
    H, W = cos_img.shape[-2], cos_img.shape[-1]
    scale = min(1.0, float(max_side) / float(max(H, W)))
    if scale < 1.0:
        newH = max(1, int(round(H * scale)))
        newW = max(1, int(round(W * scale)))
        cos_img = F.interpolate(cos_img.unsqueeze(0), size=(newH, newW), mode="bilinear", align_corners=False).squeeze(0)
        l2_img  = F.interpolate(l2_img.unsqueeze(0),  size=(newH, newW), mode="bilinear", align_corners=False).squeeze(0)

    # Composite RGB: [cos, l2, 0]
    zero = torch.zeros_like(cos_img)
    rgb = torch.cat([cos_img, l2_img, zero], dim=0).clamp(0, 1)  # (3,H,W)

    # Log images
    tb_writer.add_image(f"{tag_prefix}/cosine_map", cos_img, global_step=step)
    tb_writer.add_image(f"{tag_prefix}/l2_map", l2_img, global_step=step)
    tb_writer.add_image(f"{tag_prefix}/cos_l2_rgb", rgb, global_step=step)

    # Log scalars for first sample
    tb_writer.add_scalar(f"{tag_prefix}/cosine_mean", cos_map.mean().item(), step)      # mean in [-1,1]
    tb_writer.add_scalar(f"{tag_prefix}/cosine_min",  cos_map.min().item(), step)
    tb_writer.add_scalar(f"{tag_prefix}/cosine_max",  cos_map.max().item(), step)
    tb_writer.add_scalar(f"{tag_prefix}/l2_mean",     l2_map.mean().item(), step)
    tb_writer.add_scalar(f"{tag_prefix}/l2_p95",      torch.quantile(l2_map.flatten(), 0.95).item(), step)




# -------------------- streaming dataset --------------------
class TarShardDataset(IterableDataset):
    """
    Streams samples from a list of .tar shards.
    Each sample has:
      {key}.img  (raw image bytes)
      {key}.pt   (torch payload dict with summary + spatial_tokens)
    """
    def __init__(self, shard_paths: List[str], shuffle_shards: bool = True, seed: int = 0):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.shuffle_shards = shuffle_shards
        self.seed = seed

    def _iter_tar(self, tar_path: str) -> Iterator[Tuple[bytes, dict]]:
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            # collect base keys that have both .img and .pt
            imgs = {m.name[:-4] for m in members if m.name.endswith(".img")}
            pts  = {m.name[:-3] for m in members if m.name.endswith(".pt")}
            keys = sorted(list(imgs.intersection(pts)))
            for k in keys:
                img_m = tf.getmember(k + ".img")
                pt_m  = tf.getmember(k + ".pt")
                img_bytes = tf.extractfile(img_m).read()
                payload = torch.load(tf.extractfile(pt_m), map_location="cpu", weights_only=False)
                yield img_bytes, payload

    def __iter__(self):
        # split shards across workers
        worker = torch.utils.data.get_worker_info()
        shard_paths = self.shard_paths

        if self.shuffle_shards:
            rng = random.Random(self.seed + (worker.id if worker else 0))
            rng.shuffle(shard_paths)

        if worker is not None:
            shard_paths = shard_paths[worker.id :: worker.num_workers]

        for sp in shard_paths:
            yield from self._iter_tar(sp)


# -------------------- model heads --------------------
class SummaryHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, x_bc):
        return self.ln2(self.fc(self.ln(x_bc)))


class SpatialHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(32, out_dim), num_channels=out_dim)

    def forward(self, x_bchw):
        return self.gn(self.conv(x_bchw))


# -------------------- losses + metrics --------------------
def l2norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine_loss(a, b):
    a = l2norm(a)
    b = l2norm(b)
    return (1.0 - (a * b).sum(dim=-1)).mean()

def mse_loss(a, b):
    return F.mse_loss(a, b)

def spatial_to_tokens(x_bdhw):
    # (B, D, H, W) -> (B, T, D)
    B, D, H, W = x_bdhw.shape
    return x_bdhw.permute(0, 2, 3, 1).reshape(B, H * W, D)

def cosine_loss_spatial_tokens(s_tokens, t_tokens):
    # both: (B, T, D)
    B, T, D = s_tokens.shape
    return cosine_loss(s_tokens.reshape(B*T, D), t_tokens.reshape(B*T, D))

def mse_loss_spatial_tokens(s_tokens, t_tokens):
    B, T, D = s_tokens.shape
    return mse_loss(s_tokens.reshape(B*T, D), t_tokens.reshape(B*T, D))




import torch
from torchvision.io import decode_jpeg

def decode_image_bytes_fast(img_bytes: bytes) -> torch.Tensor:
    # returns uint8 tensor (C,H,W)
    buf = torch.frombuffer(img_bytes, dtype=torch.uint8)
    img = decode_jpeg(buf, device="cpu", mode=torchvision.io.ImageReadMode.RGB)  # (C,H,W) uint8
    return img


# -------------------- train/eval loops --------------------
@torch.no_grad()
def evaluate(student, sum_head, sp_head, dl, device, amp, Dt, Ht, Wt, tb_writer, step, lambda_mse, lambda_summary, lambda_spatial, max_batches=None):
    student.eval(); sum_head.eval(); sp_head.eval()

    agg = {"loss": 0.0, "loss_sum": 0.0, "loss_sp": 0.0,
           "cos_sum": 0.0, "cos_sp": 0.0,
           "mse_sum": 0.0, "mse_sp": 0.0}
    n = 0

    for bi, (x, t_summary, t_spatial_tokens) in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device, non_blocking=True).float()
        t_summary = t_summary.to(device, non_blocking=True).float()
        t_spatial_tokens = t_spatial_tokens.to(device, non_blocking=True).float()

        with torch.amp.autocast("cuda", enabled=amp):
            feat = student(x)[0]  # (B, Cs, Hs, Ws)
            s_sp = sp_head(feat)  # (B, Dt, Hs, Ws)
            s_sp = F.interpolate(s_sp, size=(Ht, Wt), mode="bilinear", align_corners=False)
            s_tokens = spatial_to_tokens(s_sp)  # (B, T, Dt)
            log_spatial_compare_first_sample(
                tb_writer=tb_writer,
                step=step,
                s_spatial_bdhw=s_sp,  # your student spatial map aligned to teacher grid
                t_tokens_btd=t_spatial_tokens,  # teacher cached tokens
                Ht=Ht,
                Wt=Wt,
                tag_prefix="eval/spatial_compare_first",
                max_side=512,
            )

            s_pool = feat.mean(dim=(-2, -1))
            s_sum = sum_head(s_pool)  # (B, Ct)

            cos_sum = cosine_loss(s_sum, t_summary)
            cos_sp = cosine_loss_spatial_tokens(s_tokens, t_spatial_tokens)

            mse_sum = mse_loss(s_sum, t_summary)
            mse_sp = mse_loss_spatial_tokens(s_tokens, t_spatial_tokens)

            loss_sum = cos_sum + lambda_mse * mse_sum
            loss_sp = cos_sp + lambda_mse * mse_sp

            loss = lambda_summary * loss_sum + lambda_spatial * loss_sp

        agg["loss"] += loss.item()
        agg["loss_sum"] += loss_sum.item()
        agg["loss_sp"] += loss_sp.item()
        agg["cos_sum"] += (1.0 - loss_sum.item())
        agg["cos_sp"] += (1.0 - loss_sp.item())
        agg["mse_sum"] += mse_sum.item()
        agg["mse_sp"] += mse_sp.item()
        n += 1
        with torch.no_grad():
            # ---- Summary metrics ----
            cos_sum_vec = _cosine_sim(s_sum, t_summary, dim=-1)  # (B,)
            tb_writer.add_scalar("val/summary_cos_mean", cos_sum_vec.mean().item(), step)
            tb_writer.add_scalar("val/summary_cos_p05", torch.quantile(cos_sum_vec, 0.05).item(), step)
            tb_writer.add_scalar("val/summary_cos_p95", torch.quantile(cos_sum_vec, 0.95).item(), step)

            mse_sum_vec = ((s_sum - t_summary) ** 2).mean(dim=-1)  # (B,)
            tb_writer.add_scalar("val/summary_mse_mean", mse_sum_vec.mean().item(), step)

            # scale & distribution drift (helps detect "direction matches but scale doesn't")
            tb_writer.add_scalar("val/summary_norm_ratio_mean",
                                 (s_sum.norm(dim=-1) / (t_summary.norm(dim=-1) + 1e-8)).mean().item(), step)
            tb_writer.add_scalar("val/summary_mean_abs_diff",
                                 (s_sum.mean(dim=0) - t_summary.mean(dim=0)).abs().mean().item(), step)
            tb_writer.add_scalar("val/summary_std_abs_diff",
                                 (s_sum.std(dim=0) - t_summary.std(dim=0)).abs().mean().item(), step)

            # retrieval-style agreement in batch (very informative, standard in representation learning)
            tb_writer.add_scalar("val/retrieval_top1", _batch_retrieval_top1(s_sum, t_summary).item(), step)
            tb_writer.add_scalar("val/retrieval_mrr", _batch_retrieval_mrr(s_sum, t_summary).item(), step)

            # CKA between student/teacher embeddings in the batch (structure match)
            tb_writer.add_scalar("val/summary_linear_cka", _centered_kernel_alignment(s_sum, t_summary).item(), step)

            # ---- Spatial metrics ----
            # Convert student map -> tokens
            s_tokens = _flatten_spatial_tokens(s_sp)  # (B,T,Dt)
            t_tokens = t_spatial_tokens  # (B,T,Dt)

            # per-token cosine
            cos_sp_tok = _cosine_sim(s_tokens, t_tokens, dim=-1)  # (B,T)
            tb_writer.add_scalar("val/spatial_cos_mean", cos_sp_tok.mean().item(), step)
            tb_writer.add_scalar("val/spatial_cos_p05", torch.quantile(cos_sp_tok.flatten(), 0.05).item(), step)
            tb_writer.add_scalar("val/spatial_cos_p95", torch.quantile(cos_sp_tok.flatten(), 0.95).item(), step)

            # per-token mse
            mse_sp_tok = ((s_tokens - t_tokens) ** 2).mean(dim=-1)  # (B,T)
            tb_writer.add_scalar("val/spatial_mse_mean", mse_sp_tok.mean().item(), step)

            # spatial norm ratio (again: direction may match, norms may not)
            s_tok_norm = s_tokens.norm(dim=-1)  # (B,T)
            t_tok_norm = t_tokens.norm(dim=-1)
            tb_writer.add_scalar("val/spatial_norm_ratio_mean",
                                 (s_tok_norm / (t_tok_norm + 1e-8)).mean().item(), step)

            # gram/style loss on spatial tokens (captures channel correlation / “texture” match)
            tb_writer.add_scalar("val/spatial_style_gram_mse", _style_gram_loss(s_tokens, t_tokens).item(), step)

            # edge/texture energy agreement (cheap proxy for “spatial detail”)
            s_energy = _spatial_energy(s_sp)  # (B,)
            # build teacher spatial map for energy (B,Dt,Ht,Wt)
            B, T, Dt = t_tokens.shape
            HtWt = int(math.sqrt(T))
            t_map = t_tokens.transpose(1, 2).contiguous().reshape(B, Dt, HtWt, HtWt)
            t_energy = _spatial_energy(t_map)
            tb_writer.add_scalar("val/spatial_energy_ratio_mean",
                                 (s_energy / (t_energy + 1e-8)).mean().item(), step)

            # optional: channel-wise correlation summary (expensive if done full; use a sample)
            # correlate mean token vector per sample
            s_meanD = s_tokens.mean(dim=1)  # (B,D)
            t_meanD = t_tokens.mean(dim=1)  # (B,D)
            corr = _corrcoef_1d(s_meanD.flatten(), t_meanD.flatten())
            tb_writer.add_scalar("val/spatial_meanD_corr", corr.item(), step)

    for k in agg:
        agg[k] /= max(n, 1)

    return agg


def make_collate(size: int = 416):
    def _collate(batch):
        xs, t_summ, t_spat = [], [], []
        for img_bytes, payload in batch:
            img_u8 = decode_image_bytes_fast(img_bytes)
            xs.append(preprocess_uint8(img_u8, size=size))
            t_summ.append(payload["summary"])
            t_spat.append(payload["spatial_tokens"])
        return torch.stack(xs), torch.stack(t_summ), torch.stack(t_spat)
    return _collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True, help="Root cache dir containing train/ val/ test/")
    ap.add_argument("--student_variant", type=str, default="small", choices=["tiny", "small"])
    ap.add_argument("--mode", type=str, default="debug", choices=["debug", "full"])
    ap.add_argument("--debug_train_shards", type=int, default=2)
    ap.add_argument("--debug_val_shards", type=int, default=1)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--patch_size", type=int, default=16)

    ap.add_argument("--batch_size", type=int, default=36)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--persistent_workers", type=bool, default=True)
    ap.add_argument("--prefetch_factor", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--wd", type=float, default=0.08)
    ap.add_argument("--amp", type=bool, default=True)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=50, help="How many val batches to eval each time")
    ap.add_argument('--exp_root', type=Path, default='/home/burplord/experiments/')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_summary", type=float, default=1.0)
    ap.add_argument("--lambda_spatial", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=0.05)  # small but effective
    ap.add_argument("--grad_checkpointing", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # shards
    train_shards = sorted(glob.glob(os.path.join(args.cache_dir, "train", "*.tar")))
    val_shards = sorted(glob.glob(os.path.join(args.cache_dir, "val", "*.tar")))
    if not train_shards or not val_shards:
        raise RuntimeError("No shards found. Expected train/*.tar and val/*.tar under cache_dir")

    batch_size = args.batch_size
    num_workers = args.num_workers
    eval_every =args.eval_every
    eval_batches = args.eval_batches
    grad_accum = args.grad_accum
    prefetch_factor = args.prefetch_factor
    persistent_workers = args.persistent_workers
    if args.mode == "debug":
        train_shards = train_shards[: args.debug_train_shards]
        val_shards = val_shards[: args.debug_val_shards]
        num_workers = 0
        eval_every = 5
        eval_batches = 5
        persistent_workers = False
        print(f"[DEBUG] Using train_shards={len(train_shards)} val_shards={len(val_shards)}")
    else:
        print(f"[FULL] Using train_shards={len(train_shards)} val_shards={len(val_shards)}")

    # pick student timm model name
    if args.student_variant == "tiny":
        student_name = "convnext_tiny.dinov3_lvd1689m"
    else:
        student_name = "convnext_small.dinov3_lvd1689m"

    # Datasets (streaming)
    train_ds = TarShardDataset(train_shards, shuffle_shards=True, seed=args.seed)
    val_ds = TarShardDataset(val_shards, shuffle_shards=False, seed=args.seed)

    # Build student backbone (features_only last stage)
    student = timm.create_model(
        student_name,
        pretrained=True,
        features_only=True,
        out_indices=(3,),
    ).to(device)
    if args.grad_checkpointing:
        if hasattr(student, "set_grad_checkpointing"):
            student.set_grad_checkpointing(True)
        elif hasattr(student, "grad_checkpointing"):
            student.grad_checkpointing = True

    # Determine teacher dims from one sample (read first tar)
    # (We stored summary/spatial fp16 in payload)
    first_payload = None
    for img_bytes, payload in train_ds:
        first_payload = payload
        break
    if first_payload is None:
        raise RuntimeError("Could not read any samples from cache.")

    Ct = int(first_payload["summary"].shape[0])
    Tt, Dt = first_payload["spatial_tokens"].shape
    Ht = args.size // args.patch_size
    Wt = args.size // args.patch_size
    assert Tt == Ht * Wt, f"Teacher token count T={Tt} != {Ht}*{Wt} (check size/patch_size)"

    # Student dims
    with torch.no_grad():
        feat = student(torch.randn(1, 3, args.size, args.size, device=device))[0]
        Cs = feat.shape[1]

    sum_head = SummaryHead(Cs, Ct).to(device)
    sp_head = SpatialHead(Cs, Dt).to(device)

    # Transform and loaders
    collate_fn = make_collate(size=args.size)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,   # HUGE if you run multiple epochs / long runs
        prefetch_factor=prefetch_factor,         # default=2; try 4 or 8
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,   # HUGE if you run multiple epochs / long runs
        prefetch_factor=prefetch_factor,         # default=2; try 4 or 8
    )

    # Optim
    params = []
    params += build_param_groups(student, args.wd)
    params += build_param_groups(sum_head, args.wd)
    params += build_param_groups(sp_head, args.wd)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # Train
    student.train(); sum_head.train(); sp_head.train()
    step = 0
    running = {"loss": 0.0, "cos_sum": 0.0, "cos_sp": 0.0, "mse_sum": 0.0, "mse_sp": 0.0, "loss_sum": 0.0, "loss_sp": 0.0}

    exp_dir = args.exp_root / f'exp_{time.strftime("%Y%m%d%H%M")}'
    tb_writer = SummaryWriter(log_dir=exp_dir / 'tb')
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = 0.0

    for ep in range(args.epochs):
        ep_t0 = time.time()
        for it, (x, t_summary, t_spatial_tokens) in tqdm(enumerate(train_dl)):
            x = x.to(device, non_blocking=True).float()
            t_summary = t_summary.to(device, non_blocking=True).float()
            t_spatial_tokens = t_spatial_tokens.to(device, non_blocking=True).float()

            with torch.amp.autocast("cuda", enabled=args.amp):
                feat = student(x)[0]  # (B, Cs, Hs, Ws)

                # spatial
                s_sp = sp_head(feat)  # (B, Dt, Hs, Ws)
                s_sp = F.interpolate(s_sp, size=(Ht, Wt), mode="bilinear", align_corners=False)
                s_tokens = spatial_to_tokens(s_sp)  # (B, T, Dt)

                # summary from pooled features
                s_pool = feat.mean(dim=(-2, -1))
                s_sum = sum_head(s_pool)

                # --- cosine losses ---
                cos_sum = cosine_loss(s_sum, t_summary)
                cos_sp = cosine_loss_spatial_tokens(s_tokens, t_spatial_tokens)

                # --- mse losses ---
                mse_sum = mse_loss(s_sum, t_summary)
                mse_sp = mse_loss_spatial_tokens(s_tokens, t_spatial_tokens)

                # --- combined losses ---
                loss_sum = cos_sum + args.lambda_mse * mse_sum
                loss_sp = cos_sp + args.lambda_mse * mse_sp

                loss = (
                               args.lambda_summary * loss_sum +
                               args.lambda_spatial * loss_sp
                       ) / grad_accum

            scaler.scale(loss).backward()

            if (it + 1) % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1

            running["loss"] += loss.item() * grad_accum
            running["cos_sum"] += cos_sum.item()
            running["cos_sp"] += cos_sp.item()
            running["mse_sum"] += mse_sum.item()
            running["mse_sp"] += mse_sp.item()
            running['loss_sum'] += loss_sum.item()
            running['loss_sp'] += loss_sp.item()

            denom = (it + 1)
            if (it + 1) % args.log_every == 0:
                print(
                    f"ep={ep} it={it} step={step} "
                    f"loss={running['loss'] / denom:.4f} "
                    f"cos_sum={running['cos_sum'] / denom:.4f} "
                    f"cos_sp={running['cos_sp'] / denom:.4f} "
                    f"mse_sum={running['mse_sum'] / denom:.6f} "
                    f"mse_sp={running['mse_sp'] / denom:.6f} "
                    f"loss_sum={running['loss_sum'] / denom:.4f} "
                    f"loss_sp={running['loss_sp'] / denom:.4f}"
                )
            tb_writer.add_scalar("train_loss", running['loss']/denom, step)
            tb_writer.add_scalar("train_cos_sum", running['cos_sum']/denom, step)
            tb_writer.add_scalar("train_cos_sp", running['cos_sp']/denom, step)
            tb_writer.add_scalar("train_mse_sum", running['mse_sum']/denom, step)
            tb_writer.add_scalar("train_mse_sp", running['mse_sp']/denom, step)
            tb_writer.add_scalar("train_loss_sum", running['loss_sum']/denom, step)
            tb_writer.add_scalar("train_loss_sp", running['loss_sp']/denom, step)
            tb_writer.add_scalar("iteration", it, step)


            # periodic eval
            if step > 0 and (step % eval_every == 0):
                print('Evaluating..')
                metrics = evaluate(
                    student, sum_head, sp_head,
                    val_dl, device=device, amp=args.amp,
                    Dt=Dt, Ht=Ht, Wt=Wt,
                    max_batches=eval_batches,
                    tb_writer=tb_writer,
                    step=step,
                    lambda_mse=args.lambda_mse,
                    lambda_spatial=args.lambda_spatial,
                    lambda_summary=args.lambda_summary,
                )
                print(
                    f"[VAL] step={step} "
                    f"loss={metrics['loss']:.4f} "
                    f"loss_sum={metrics['loss_sum']:.4f} loss_sp={metrics['loss_sp']:.4f}"
                    f"cos_sum={metrics['cos_sum']:.4f} cos_sp={metrics['cos_sp']:.4f} "
                    f"mse_sum={metrics['mse_sum']:.6f} mse_sp={metrics['mse_sp']:.6f}"
                )
                tb_writer.add_scalar("val_loss", metrics['loss'], step)
                tb_writer.add_scalar("val_sum_loss", metrics['loss_sum'], step)
                tb_writer.add_scalar("val_sp_loss", metrics['loss_sp'], step)
                tb_writer.add_scalar("val_cos_sum", metrics['cos_sum'], step)
                tb_writer.add_scalar("val_cos_sp", metrics['cos_sp'], step)
                tb_writer.add_scalar("val_mse_sum", metrics['mse_sum'], step)
                tb_writer.add_scalar("val_mse_sp", metrics['mse_sp'], step)

                if best_val_loss > metrics['loss']:
                    best_val_loss = metrics['loss']
                    save_path = checkpoint_dir / f'epoch_{ep}_step_{step}_val_loss_{metrics['loss']:.3f}.pth'
                    torch.save(
                        {
                            "student_name": student_name,
                            "size": args.size,
                            "patch_size": args.patch_size,
                            "teacher_summary_dim": Ct,
                            "teacher_spatial_dim": Dt,
                            "student_state_dict": student.state_dict(),
                            "sum_head_state_dict": sum_head.state_dict(),
                            "sp_head_state_dict": sp_head.state_dict(),
                        },
                        save_path,
                    )
                    print(f"Saved checkpoint: {save_path}, Updated Best Val: {best_val_loss}")

        tb_writer.add_scalar("epoch", ep, step)

        print(f"Epoch {ep} done in {(time.time()-ep_t0)/60:.1f} min")


if __name__ == "__main__":
    main()