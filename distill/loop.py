#!/usr/bin/env python3
# distill_convnext_from_cache_refactor.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from distill.model import cosine_loss, cosine_loss_spatial_tokens, mse_loss, mse_loss_spatial_tokens, DistillModel, \
    log_spatial_compare_first_sample, _cosine_sim, _batch_retrieval_top1, _batch_retrieval_mrr, \
    _centered_kernel_alignment, _flatten_spatial_tokens, _style_gram_loss, _spatial_energy, _corrcoef_1d


# ============================================================
# TRAIN / EVAL LOOPS
# ============================================================

@dataclass
class LossWeights:
    lambda_summary: float = 1.0
    lambda_spatial: float = 1.0
    lambda_mse: float = 0.05


def compute_losses(
        s_sum: torch.Tensor,
        s_tokens: torch.Tensor,
        t_summary: torch.Tensor,
        t_tokens: torch.Tensor,
        w: LossWeights,
) -> Dict[str, torch.Tensor]:
    cos_sum = cosine_loss(s_sum, t_summary)
    cos_sp = cosine_loss_spatial_tokens(s_tokens, t_tokens)

    mse_sum = mse_loss(s_sum, t_summary)
    mse_sp = mse_loss_spatial_tokens(s_tokens, t_tokens)

    loss_sum = cos_sum + w.lambda_mse * mse_sum
    loss_sp = cos_sp + w.lambda_mse * mse_sp
    loss = w.lambda_summary * loss_sum + w.lambda_spatial * loss_sp

    return {
        "loss": loss,
        "loss_sum": loss_sum,
        "loss_sp": loss_sp,
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
        max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    agg = {
        "loss": 0.0, "loss_sum": 0.0, "loss_sp": 0.0,
        "cos_sum": 0.0, "cos_sp": 0.0,
        "mse_sum": 0.0, "mse_sp": 0.0
    }
    n = 0

    for bi, (x, t_summary, t_spatial_tokens) in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device, non_blocking=True).float()
        t_summary = t_summary.to(device, non_blocking=True).float()
        t_spatial_tokens = t_spatial_tokens.to(device, non_blocking=True).float()

        with torch.amp.autocast("cuda", enabled=amp):
            s_sum, s_tokens, s_sp, _ = model(x)
            out = compute_losses(s_sum, s_tokens, t_summary, t_spatial_tokens, w)

            if bi % 3 == 0 and bi < 15:
                # hotspot inspect + image logs
                s0 = s_sp[0]  # (Dt,Ht,Wt)
                t0 = t_spatial_tokens[0].transpose(0, 1).contiguous().reshape(Dt, Ht, Wt)

                l2 = torch.sqrt(((s0 - t0) ** 2).sum(dim=0) + 1e-8)
                flat = l2.flatten()
                k = flat.argmax().item()
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
                    "t_norm max/p95/mean:", t_norm.max().item(), torch.quantile(t_norm, 0.95).item(),
                    t_norm.mean().item(),
                    "s_norm max/p95/mean:", s_norm.max().item(), torch.quantile(s_norm, 0.95).item(),
                    s_norm.mean().item(),
                )
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

        agg["loss"] += out["loss"].item()
        agg["loss_sum"] += out["loss_sum"].item()
        agg["loss_sp"] += out["loss_sp"].item()
        agg["cos_sum"] += (1.0 - out["cos_sum"].item())
        agg["cos_sp"] += (1.0 - out["cos_sp"].item())
        agg["mse_sum"] += out["mse_sum"].item()
        agg["mse_sp"] += out["mse_sp"].item()
        n += 1

        # TB (extra diagnostics)
        with torch.no_grad():
            cos_sum_vec = _cosine_sim(s_sum, t_summary, dim=-1)
            tb_writer.add_scalar("val/summary_cos_mean", cos_sum_vec.mean().item(), step)
            tb_writer.add_scalar("val/summary_cos_p05", torch.quantile(cos_sum_vec, 0.05).item(), step)
            tb_writer.add_scalar("val/summary_cos_p95", torch.quantile(cos_sum_vec, 0.95).item(), step)

            mse_sum_vec = ((s_sum - t_summary) ** 2).mean(dim=-1)
            tb_writer.add_scalar("val/summary_mse_mean", mse_sum_vec.mean().item(), step)

            tb_writer.add_scalar(
                "val/summary_norm_ratio_mean",
                (s_sum.norm(dim=-1) / (t_summary.norm(dim=-1) + 1e-8)).mean().item(),
                step,
            )
            tb_writer.add_scalar(
                "val/summary_mean_abs_diff",
                (s_sum.mean(dim=0) - t_summary.mean(dim=0)).abs().mean().item(),
                step,
            )
            tb_writer.add_scalar(
                "val/summary_std_abs_diff",
                (s_sum.std(dim=0) - t_summary.std(dim=0)).abs().mean().item(),
                step,
            )

            tb_writer.add_scalar("val/retrieval_top1", _batch_retrieval_top1(s_sum, t_summary).item(), step)
            tb_writer.add_scalar("val/retrieval_mrr", _batch_retrieval_mrr(s_sum, t_summary).item(), step)
            tb_writer.add_scalar("val/summary_linear_cka", _centered_kernel_alignment(s_sum, t_summary).item(), step)

            s_tokens2 = _flatten_spatial_tokens(s_sp)
            t_tokens2 = t_spatial_tokens

            cos_sp_tok = _cosine_sim(s_tokens2, t_tokens2, dim=-1)
            tb_writer.add_scalar("val/spatial_cos_mean", cos_sp_tok.mean().item(), step)
            tb_writer.add_scalar("val/spatial_cos_p05", torch.quantile(cos_sp_tok.flatten(), 0.05).item(), step)
            tb_writer.add_scalar("val/spatial_cos_p95", torch.quantile(cos_sp_tok.flatten(), 0.95).item(), step)

            mse_sp_tok = ((s_tokens2 - t_tokens2) ** 2).mean(dim=-1)
            tb_writer.add_scalar("val/spatial_mse_mean", mse_sp_tok.mean().item(), step)

            s_tok_norm = s_tokens2.norm(dim=-1)
            t_tok_norm = t_tokens2.norm(dim=-1)
            tb_writer.add_scalar("val/spatial_norm_ratio_mean", (s_tok_norm / (t_tok_norm + 1e-8)).mean().item(), step)

            tb_writer.add_scalar("val/spatial_style_gram_mse", _style_gram_loss(s_tokens2, t_tokens2).item(), step)

            s_energy = _spatial_energy(s_sp)
            B, Tt, Dt_ = t_tokens2.shape
            t_map = t_tokens2.transpose(1, 2).contiguous().reshape(B, Dt_, Ht, Wt)
            t_energy = _spatial_energy(t_map)
            tb_writer.add_scalar("val/spatial_energy_ratio_mean", (s_energy / (t_energy + 1e-8)).mean().item(), step)

            s_meanD = s_tokens2.mean(dim=1)
            t_meanD = t_tokens2.mean(dim=1)
            corr = _corrcoef_1d(s_meanD.flatten(), t_meanD.flatten())
            tb_writer.add_scalar("val/spatial_meanD_corr", corr.item(), step)

    for k in agg:
        agg[k] /= max(n, 1)

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
) -> Tuple[int, Optional[float]]:
    model.train()

    step = step0
    running = {
        "loss": 0.0, "cos_sum": 0.0, "cos_sp": 0.0,
        "mse_sum": 0.0, "mse_sp": 0.0,
        "loss_sum": 0.0, "loss_sp": 0.0,
    }

    for it, (x, t_summary, t_spatial_tokens) in tqdm(enumerate(train_dl), total=None):
        x = x.to(device, non_blocking=True).float()
        t_summary = t_summary.to(device, non_blocking=True).float()
        t_spatial_tokens = t_spatial_tokens.to(device, non_blocking=True).float()

        with torch.amp.autocast("cuda", enabled=amp):
            s_sum, s_tokens, s_sp, _ = model(x)
            out = compute_losses(s_sum, s_tokens, t_summary, t_spatial_tokens, w)
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
            )

        # TB train
        tb_writer.add_scalar("train_loss", running["loss"] / denom, step)
        tb_writer.add_scalar("train_cos_sum", running["cos_sum"] / denom, step)
        tb_writer.add_scalar("train_cos_sp", running["cos_sp"] / denom, step)
        tb_writer.add_scalar("train_mse_sum", running["mse_sum"] / denom, step)
        tb_writer.add_scalar("train_mse_sp", running["mse_sp"] / denom, step)
        tb_writer.add_scalar("train_loss_sum", running["loss_sum"] / denom, step)
        tb_writer.add_scalar("train_loss_sp", running["loss_sp"] / denom, step)
        tb_writer.add_scalar("iteration", it, step)

        # periodic eval
        if step > 0 and (step % eval_every == 0):
            print("Evaluating..")
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
            )
            print(
                f"[VAL] step={step} "
                f"loss={metrics['loss']:.4f} "
                f"loss_sum={metrics['loss_sum']:.4f} loss_sp={metrics['loss_sp']:.4f} "
                f"cos_sum={metrics['cos_sum']:.4f} cos_sp={metrics['cos_sp']:.4f} "
                f"mse_sum={metrics['mse_sum']:.6f} mse_sp={metrics['mse_sp']:.6f}"
            )

            tb_writer.add_scalar("val_loss", metrics["loss"], step)
            tb_writer.add_scalar("val_sum_loss", metrics["loss_sum"], step)
            tb_writer.add_scalar("val_sp_loss", metrics["loss_sp"], step)
            tb_writer.add_scalar("val_cos_sum", metrics["cos_sum"], step)
            tb_writer.add_scalar("val_cos_sp", metrics["cos_sp"], step)
            tb_writer.add_scalar("val_mse_sum", metrics["mse_sum"], step)
            tb_writer.add_scalar("val_mse_sp", metrics["mse_sp"], step)

            if (best_val_loss is None) or (best_val_loss > metrics["loss"]):
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

    return step, best_val_loss
