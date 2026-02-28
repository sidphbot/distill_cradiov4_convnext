#!/usr/bin/env python3
# distill_convnext_from_cache_refactor.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from distill.model import cosine_loss, cosine_loss_spatial_tokens, mse_loss, DistillModel, \
    log_spatial_compare_first_sample, _cosine_sim, _batch_retrieval_top1, _batch_retrieval_mrr, \
    _centered_kernel_alignment, _flatten_spatial_tokens, _style_gram_loss, _spatial_energy, _corrcoef_1d, \
    sobel_grad_loss

# ============================================================
# TRAIN / EVAL LOOPS
# ============================================================

# loop.py
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def _pearson_corr_per_sample(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    a,b: (B, N) -> returns (B,) Pearson correlation
    """
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = (a.norm(dim=1) * b.norm(dim=1) + eps)
    return num / den


def _sobel_mag_2d(x_b1hw: torch.Tensor) -> torch.Tensor:
    """
    x_b1hw: (B,1,H,W) -> returns (B,1,H,W) gradient magnitude (Sobel)
    """
    device, dtype = x_b1hw.device, x_b1hw.dtype
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=device, dtype=dtype) / 8.0
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=device, dtype=dtype) / 8.0
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)

    gx = F.conv2d(x_b1hw, kx, padding=1)
    gy = F.conv2d(x_b1hw, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def _laplacian_highpass_depthwise(x_bchw: torch.Tensor) -> torch.Tensor:
    """
    x_bchw: (B,C,H,W) -> returns (B,C,H,W) Laplacian high-pass per channel (depthwise)
    """
    B, C, H, W = x_bchw.shape
    device, dtype = x_bchw.device, x_bchw.dtype
    k = torch.tensor([[0., 1., 0.],
                      [1., -4., 1.],
                      [0., 1., 0.]], device=device, dtype=dtype) / 4.0
    k = k.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # (C,1,3,3)
    return F.conv2d(x_bchw, k, padding=1, groups=C)

def hb(tag):
    torch.cuda.synchronize()
    print(f"[HB] {tag}", flush=True)


def tokens_to_map(x_tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
    # Example: tokens are (B, HW, C). Adjust if yours is (B,C,HW).
    B, HW, C = x_tokens.shape
    assert HW == H * W
    return x_tokens.transpose(1, 2).contiguous().view(B, C, H, W)


def ramp_linear(step: int, warmup_steps: int, start: float, end: float) -> float:
    """Linear ramp from start->end over warmup_steps, then stays at end."""
    if warmup_steps <= 0:
        return end
    t = min(max(step, 0), warmup_steps) / float(warmup_steps)
    return start + t * (end - start)


@dataclass
class LossWeights:
    lambda_summary: float = 1.0
    lambda_spatial: float = 1.0
    lambda_mse: float = 0.05
    lambda_sp_mse: float = 0.5
    lambda_grad: float = 0.05


def compute_losses(
        s_sum: torch.Tensor,
        s_tokens: torch.Tensor,
        s_sp: torch.Tensor,
        t_summary: torch.Tensor,
        t_tokens: torch.Tensor,
        w: LossWeights,
        grad_eps: float,
        Ht: int,
        Wt: int,
) -> Dict[str, torch.Tensor]:
    cos_sum = cosine_loss(s_sum, t_summary)
    cos_sp = cosine_loss_spatial_tokens(s_tokens, t_tokens)

    t_sp = tokens_to_map(t_tokens, Ht, Wt)

    mse_sum = mse_loss(s_sum, t_summary)
    mse_sp = mse_loss(s_sp, t_sp)

    grad_loss = sobel_grad_loss(s_sp, t_sp, eps=grad_eps)

    loss_sum = cos_sum + w.lambda_mse * mse_sum
    loss_sp = cos_sp + w.lambda_sp_mse * mse_sp + w.lambda_grad * grad_loss
    loss = w.lambda_summary * loss_sum + w.lambda_spatial * loss_sp

    return {
        "loss": loss,
        "loss_sum": loss_sum,
        "loss_sp": loss_sp,
        "loss_grad": grad_loss,
        "cos_sum": cos_sum,
        "cos_sp": cos_sp,
        "mse_sum": mse_sum,
        "mse_sp": mse_sp,
    }


@torch.no_grad()
def evaluate(
        model: DistillModel,
        dl: DataLoader,
        device: str,
        amp: bool,
        Dt: int,
        Ht: int,
        Wt: int,
        tb_writer: SummaryWriter,
        step: int,
        w: LossWeights,
        grad_eps: float,
        max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    agg = {
        "loss": 0.0, "loss_sum": 0.0, "loss_sp": 0.0,
        "cos_sum": 0.0, "cos_sp": 0.0,
        "mse_sum": 0.0, "mse_sp": 0.0, "loss_grad": 0.0,
    }
    n = 0

    # ---- log-once accumulators (CPU tensors for quantiles) ----
    summary_cos_all = []
    spatial_cos_all = []
    edge_corr_all = []

    # ---- scalar accumulators ----
    retrieval_top1_sum = 0.0
    retrieval_mrr_sum = 0.0
    summary_cka_sum = 0.0

    summary_mse_mean_sum = 0.0
    summary_norm_ratio_sum = 0.0
    summary_mean_abs_diff_sum = 0.0
    summary_std_abs_diff_sum = 0.0

    spatial_mse_mean_sum = 0.0
    spatial_norm_ratio_sum = 0.0
    spatial_style_gram_sum = 0.0
    spatial_energy_ratio_sum = 0.0
    spatial_meanD_corr_sum = 0.0

    hf_cos_sum = 0.0
    hf_mse_sum = 0.0
    hf_energy_ratio_sum = 0.0

    for bi, batch in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break

        # Support both (x,t_sum,t_sp) and (x,t_sum,t_sp,extra)
        if len(batch) == 3:
            x, t_summary, t_spatial_tokens = batch
        else:
            x, t_summary, t_spatial_tokens, _ = batch

        x = x.to(device, non_blocking=True).float()
        t_summary = t_summary.to(device, non_blocking=True).float()
        t_spatial_tokens = t_spatial_tokens.to(device, non_blocking=True).float()

        with torch.amp.autocast("cuda", enabled=amp):
            s_sum, s_tokens, s_sp, _ = model(x)
            out = compute_losses(s_sum, s_tokens, s_sp, t_summary, t_spatial_tokens, w, grad_eps, Ht, Wt)

            if bi % 3 == 0 and bi < 15:
                # hotspot inspect + image logs (fine to keep)
                s0 = s_sp[0]
                t0 = t_spatial_tokens[0].transpose(0, 1).contiguous().reshape(Dt, Ht, Wt)

                l2 = torch.sqrt(((s0 - t0) ** 2).sum(dim=0) + 1e-8)
                k = l2.flatten().argmax().item()
                hh, ww = divmod(k, Wt)

                s_vec = s0[:, hh, ww]
                t_vec = t0[:, hh, ww]
                s_norm = s_tokens[0].norm(dim=-1)
                t_norm = t_spatial_tokens[0].norm(dim=-1)
                print(
                    "HOT (h,w)=", (hh, ww),
                    "l2=", l2[hh, ww].item(),
                    "cos=", F.cosine_similarity(s_vec, t_vec, dim=0).item(),
                    "||s||=", s_vec.norm().item(),
                    "||t||=", t_vec.norm().item(),
                    "t_norm max/p95/mean:", t_norm.max().item(), torch.quantile(t_norm, 0.95).item(), t_norm.mean().item(),
                    "s_norm max/p95/mean:", s_norm.max().item(), torch.quantile(s_norm, 0.95).item(), s_norm.mean().item(),
                )

                if tb_writer is not None:
                    log_spatial_compare_first_sample(
                        tb_writer=tb_writer,
                        step=step,
                        s_spatial_bdhw=s_sp,
                        t_tokens_btd=t_spatial_tokens,
                        Ht=Ht,
                        Wt=Wt,
                        tag_prefix=f"eval/spatial_compare_{bi}_first",
                        max_side=512,
                    )

        # ---- aggregate losses ----
        agg["loss"] += out["loss"].item()
        agg["loss_sum"] += out["loss_sum"].item()
        agg["loss_sp"] += out["loss_sp"].item()
        agg["loss_grad"] += out["loss_grad"].item()
        agg["cos_sum"] += (1.0 - out["cos_sum"].item())
        agg["cos_sp"] += (1.0 - out["cos_sp"].item())
        agg["mse_sum"] += out["mse_sum"].item()
        agg["mse_sp"] += out["mse_sp"].item()
        n += 1

        # ---- build shared tensors for diagnostics ----
        # tokens2: (B,T,Dt), teacher map: (B,Dt,Ht,Wt)
        s_tokens2 = _flatten_spatial_tokens(s_sp)
        t_tokens2 = t_spatial_tokens
        B, Tt, Dt_ = t_tokens2.shape
        t_map = t_tokens2.transpose(1, 2).contiguous().reshape(B, Dt_, Ht, Wt)

        # ---- cosine vectors for quantiles (CPU) ----
        summary_cos = _cosine_sim(s_sum, t_summary, dim=-1).detach().float().cpu()  # (B,)
        summary_cos_all.append(summary_cos)

        spatial_cos = _cosine_sim(s_tokens2, t_tokens2, dim=-1).detach().float().cpu()  # (B,T)
        spatial_cos_all.append(spatial_cos.flatten())  # collect as 1D

        # ---- other scalar diagnostics ----
        retrieval_top1_sum += float(_batch_retrieval_top1(s_sum, t_summary).detach().cpu())
        retrieval_mrr_sum += float(_batch_retrieval_mrr(s_sum, t_summary).detach().cpu())
        summary_cka_sum += float(_centered_kernel_alignment(s_sum, t_summary).detach().cpu())

        summary_mse_mean_sum += float(((s_sum - t_summary) ** 2).mean(dim=-1).mean().detach().cpu())
        summary_norm_ratio_sum += float((s_sum.norm(dim=-1) / (t_summary.norm(dim=-1) + 1e-8)).mean().detach().cpu())
        summary_mean_abs_diff_sum += float((s_sum.mean(dim=0) - t_summary.mean(dim=0)).abs().mean().detach().cpu())
        summary_std_abs_diff_sum += float((s_sum.std(dim=0) - t_summary.std(dim=0)).abs().mean().detach().cpu())

        spatial_mse_mean_sum += float(((s_tokens2 - t_tokens2) ** 2).mean(dim=-1).mean().detach().cpu())
        spatial_norm_ratio_sum += float((s_tokens2.norm(dim=-1) / (t_tokens2.norm(dim=-1) + 1e-8)).mean().detach().cpu())
        spatial_style_gram_sum += float(_style_gram_loss(s_tokens2, t_tokens2).detach().cpu())

        s_energy = _spatial_energy(s_sp)
        t_energy = _spatial_energy(t_map)
        spatial_energy_ratio_sum += float((s_energy / (t_energy + 1e-8)).mean().detach().cpu())

        s_meanD = s_tokens2.mean(dim=1)
        t_meanD = t_tokens2.mean(dim=1)
        spatial_meanD_corr_sum += float(_corrcoef_1d(s_meanD.flatten(), t_meanD.flatten()).detach().cpu())

        # ---- Edge alignment (accumulate only) ----
        x_gray = x.mean(dim=1, keepdim=True)     # (B,1,H,W)
        g_img = _sobel_mag_2d(x_gray)            # (B,1,H,W)

        s_energy_map = torch.sqrt((s_sp * s_sp).sum(dim=1, keepdim=True) + 1e-8)  # (B,1,Ht,Wt)
        g_feat = _sobel_mag_2d(s_energy_map)     # (B,1,Ht,Wt)

        g_img_ds = F.interpolate(g_img, size=(Ht, Wt), mode="bilinear", align_corners=False)

        corr_b = _pearson_corr_per_sample(g_feat.flatten(1), g_img_ds.flatten(1))  # (B,)
        edge_corr_all.append(corr_b.detach().float().cpu())

        # ---- High-frequency match (accumulate only) ----
        s_hf = _laplacian_highpass_depthwise(s_sp)
        t_hf = _laplacian_highpass_depthwise(t_map)

        s_hf_tok = _flatten_spatial_tokens(s_hf)
        t_hf_tok = _flatten_spatial_tokens(t_hf)

        hf_cos_sum += float(_cosine_sim(s_hf_tok, t_hf_tok, dim=-1).mean().detach().cpu())
        hf_mse_sum += float(F.mse_loss(s_hf, t_hf).detach().cpu())

        s_hf_energy = _spatial_energy(s_hf)
        t_hf_energy = _spatial_energy(t_hf)
        hf_energy_ratio_sum += float((s_hf_energy / (t_hf_energy + 1e-8)).mean().detach().cpu())

    # ---- finalize mean losses ----
    for k in agg:
        agg[k] /= max(n, 1)

    # ---- log once ----
    if tb_writer is not None and n > 0:
        # summary cosine quantiles
        summary_cos_cat = torch.cat(summary_cos_all, dim=0)
        tb_writer.add_scalar("val/summary_cos_mean", summary_cos_cat.mean().item(), step)
        tb_writer.add_scalar("val/summary_cos_p05", torch.quantile(summary_cos_cat, 0.05).item(), step)
        tb_writer.add_scalar("val/summary_cos_p95", torch.quantile(summary_cos_cat, 0.95).item(), step)

        # spatial cosine quantiles
        spatial_cos_cat = torch.cat(spatial_cos_all, dim=0)
        tb_writer.add_scalar("val/spatial_cos_mean", spatial_cos_cat.mean().item(), step)
        tb_writer.add_scalar("val/spatial_cos_p05", torch.quantile(spatial_cos_cat, 0.05).item(), step)
        tb_writer.add_scalar("val/spatial_cos_p95", torch.quantile(spatial_cos_cat, 0.95).item(), step)

        # other scalars (means over batches)
        tb_writer.add_scalar("val/retrieval_top1", retrieval_top1_sum / n, step)
        tb_writer.add_scalar("val/retrieval_mrr", retrieval_mrr_sum / n, step)
        tb_writer.add_scalar("val/summary_linear_cka", summary_cka_sum / n, step)

        tb_writer.add_scalar("val/summary_mse_mean", summary_mse_mean_sum / n, step)
        tb_writer.add_scalar("val/summary_norm_ratio_mean", summary_norm_ratio_sum / n, step)
        tb_writer.add_scalar("val/summary_mean_abs_diff", summary_mean_abs_diff_sum / n, step)
        tb_writer.add_scalar("val/summary_std_abs_diff", summary_std_abs_diff_sum / n, step)

        tb_writer.add_scalar("val/spatial_mse_mean", spatial_mse_mean_sum / n, step)
        tb_writer.add_scalar("val/spatial_norm_ratio_mean", spatial_norm_ratio_sum / n, step)
        tb_writer.add_scalar("val/spatial_style_gram_mse", spatial_style_gram_sum / n, step)
        tb_writer.add_scalar("val/spatial_energy_ratio_mean", spatial_energy_ratio_sum / n, step)
        tb_writer.add_scalar("val/spatial_meanD_corr", spatial_meanD_corr_sum / n, step)

        # edge alignment quantiles
        edge_corr_cat = torch.cat(edge_corr_all, dim=0)
        tb_writer.add_scalar("val/edge_align_corr_mean", edge_corr_cat.mean().item(), step)
        tb_writer.add_scalar("val/edge_align_corr_p05", torch.quantile(edge_corr_cat, 0.05).item(), step)
        tb_writer.add_scalar("val/edge_align_corr_p95", torch.quantile(edge_corr_cat, 0.95).item(), step)

        # HF metrics (means over batches)
        tb_writer.add_scalar("val/hf_cos_mean", hf_cos_sum / n, step)
        tb_writer.add_scalar("val/hf_mse_mean", hf_mse_sum / n, step)
        tb_writer.add_scalar("val/hf_energy_ratio_mean", hf_energy_ratio_sum / n, step)

    return agg


def train_one_epoch(
        model: DistillModel,
        train_dl: DataLoader,
        val_dl: DataLoader,
        device: str,
        opt: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        tb_writer: SummaryWriter,
        step0: int,
        ep: int,
        *,
        amp: bool,
        grad_accum: int,
        log_every: int,
        eval_every: int,
        eval_batches: int,
        Dt: int,
        Ht: int,
        Wt: int,
        w: LossWeights,
        checkpoint_dir: Path,
        student_name: str,
        size: int,
        patch_size: int,
        best_val_loss: Optional[float],
        mse_sp_w_start: float,
        mse_sp_w_end: float,
        mse_sp_w_warmup_frac: float,
        grad_w_start: float,
        grad_w_end: float,
        grad_w_warmup_frac: float,
        grad_eps: float = 1e-3,
) -> Tuple[int, Optional[float]]:
    model.train()

    step = step0
    running = {
        "loss": 0.0, "cos_sum": 0.0, "cos_sp": 0.0,
        "mse_sum": 0.0, "mse_sp": 0.0,
        "loss_sum": 0.0, "loss_sp": 0.0, 'loss_grad': 0.0
    }

    for it, (x, t_summary, t_spatial_tokens, keys) in tqdm(enumerate(train_dl), total=None):
        steps_per_epoch = 3800

        w.lambda_sp_mse = ramp_linear(step=step, warmup_steps=int(mse_sp_w_warmup_frac * steps_per_epoch),
                                      start=mse_sp_w_start, end=mse_sp_w_end)

        w.lambda_grad = ramp_linear(step=step, warmup_steps=int(grad_w_warmup_frac * steps_per_epoch),
                                    start=grad_w_start, end=grad_w_end)

        x = x.to(device, non_blocking=True).float()
        t_summary = t_summary.to(device, non_blocking=True).float()
        t_spatial_tokens = t_spatial_tokens.to(device, non_blocking=True).float()

        with torch.amp.autocast("cuda", enabled=amp):
            s_sum, s_tokens, s_sp, _ = model(x)
            out = compute_losses(s_sum, s_tokens, s_sp, t_summary, t_spatial_tokens, w, grad_eps, Ht, Wt)
            loss = out["loss"] / grad_accum

        scaler.scale(loss).backward()

        if (it + 1) % grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1

        # running stats
        running["loss"] += loss.item() * grad_accum
        running["cos_sum"] += out["cos_sum"].item()
        running["cos_sp"] += out["cos_sp"].item()
        running["mse_sum"] += out["mse_sum"].item()
        running["mse_sp"] += out["mse_sp"].item()
        running["loss_sum"] += out["loss_sum"].item()
        running["loss_sp"] += out["loss_sp"].item()
        running["loss_grad"] += out["loss_grad"].item()

        denom = (it + 1)

        if (it + 1) % log_every == 0:
            print(
                f"ep={ep} it={it} step={step} "
                f"loss={running['loss'] / denom:.4f} "
                f"cos_sum={running['cos_sum'] / denom:.4f} "
                f"cos_sp={running['cos_sp'] / denom:.4f} "
                f"mse_sum={running['mse_sum'] / denom:.6f} "
                f"mse_sp={running['mse_sp'] / denom:.6f} " 
                f"loss_sum={running['loss_sum'] / denom:.4f} "
                f"loss_sp={running['loss_sp'] / denom:.4f}"
                f"loss_grad={running['loss_grad'] / denom:.4f}"
                f"mse_sp_w={w.lambda_sp_mse:.4f}"
                f"grad_w={w.lambda_grad:.4f}"
            )

        # TB train
        tb_writer.add_scalar("train_loss", running["loss"] / denom, step)
        tb_writer.add_scalar("train_cos_sum", running["cos_sum"] / denom, step)
        tb_writer.add_scalar("train_cos_sp", running["cos_sp"] / denom, step)
        tb_writer.add_scalar("train_mse_sum", running["mse_sum"] / denom, step)
        tb_writer.add_scalar("train_mse_sp", running["mse_sp"] / denom, step)
        tb_writer.add_scalar("train_loss_sum", running["loss_sum"] / denom, step)
        tb_writer.add_scalar("train_loss_sp", running["loss_sp"] / denom, step)
        tb_writer.add_scalar("train_loss_grad", running["loss_grad"] / denom, step)
        tb_writer.add_scalar("iteration", it, step)
        tb_writer.add_scalar("train/mse_sp_w", w.lambda_sp_mse, step)
        tb_writer.add_scalar("train/grad_w", w.lambda_grad, step)

        # periodic eval
        if step > 0 and (step % eval_every == 0):
            print("Evaluating..")
            best_val_loss = evaluate_step(Dt, Ht, Wt, amp, best_val_loss, checkpoint_dir, device, ep, eval_batches,
                                          grad_eps, model, w.lambda_sp_mse, patch_size, size, step, student_name, tb_writer,
                                          val_dl, w)

    best_val_loss = evaluate_step(Dt, Ht, Wt, amp, best_val_loss, checkpoint_dir, device, ep, eval_batches,
                                  grad_eps, model, w.lambda_sp_mse, patch_size, size, step, student_name, tb_writer,
                                  val_dl, w, force_save=True)

    return step, best_val_loss


def evaluate_step(Dt: int, Ht: int, Wt: int, amp: bool, best_val_loss: float | None, checkpoint_dir: Path, device: str,
                  ep: int, eval_batches: int, grad_eps: float, model: DistillModel, mse_sp_w: float, patch_size: int,
                  size: int, step: int, student_name: str, tb_writer: SummaryWriter, val_dl: DataLoader,
                  w: LossWeights, force_save: bool = False) -> float:
    metrics = evaluate(
        model=model,
        dl=val_dl,
        device=device,
        amp=amp,
        Dt=Dt,
        Ht=Ht,
        Wt=Wt,
        tb_writer=tb_writer,
        step=step,
        w=w,
        max_batches=eval_batches,
        grad_eps=grad_eps
    )
    print(
        f"[VAL] step={step} "
        f"loss={metrics['loss']:.4f} "
        f"loss_sum={metrics['loss_sum']:.4f} loss_sp={metrics['loss_sp']:.4f} "
        f"cos_sum={metrics['cos_sum']:.4f} cos_sp={metrics['cos_sp']:.4f} "
        f"mse_sum={metrics['mse_sum']:.6f} mse_sp={metrics['mse_sp']:.6f}"
        f"loss_grad={metrics['loss_grad']:.6f} mse_sp_w={mse_sp_w:.6f}"
    )

    tb_writer.add_scalar("val_loss", metrics["loss"], step)
    tb_writer.add_scalar("val_loss_sum", metrics["loss_sum"], step)
    tb_writer.add_scalar("val_loss_sp", metrics["loss_sp"], step)
    tb_writer.add_scalar("val_loss_grad", metrics["loss_grad"], step)
    tb_writer.add_scalar("val_cos_sum", metrics["cos_sum"], step)
    tb_writer.add_scalar("val_cos_sp", metrics["cos_sp"], step)
    tb_writer.add_scalar("val_mse_sum", metrics["mse_sum"], step)
    tb_writer.add_scalar("val_mse_sp", metrics["mse_sp"], step)

    if (best_val_loss is None) or (best_val_loss > metrics["loss"]) or force_save:
        best_val_loss = metrics["loss"]
        save_path = checkpoint_dir / f"epoch_{ep}_step_{step}_val_loss_{metrics['loss']:.3f}.pth"
        torch.save(
            {
                "student_name": student_name,
                "size": size,
                "patch_size": patch_size,
                "teacher_summary_dim": model.sum_head.fc.out_features,
                "teacher_spatial_dim": model.sp_head.conv.out_channels,
                "student_state_dict": model.student.state_dict(),
                "sum_head_state_dict": model.sum_head.state_dict(),
                "sp_head_state_dict": model.sp_head.state_dict(),
            },
            save_path,
        )
        print(f"Saved checkpoint: {save_path}, Updated Best Val: {best_val_loss}")
    return best_val_loss
