"""PyTorch Lightning module for distillation training."""

from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid

from distill.augment import apply_student_augmentations, build_augmentation_pipeline
from distill.muon import build_adamw_muon_optimizer
from distill.model import (
    DistillModel,
    LossWeights,
    compute_losses,
    cosine_loss,
    cosine_loss_spatial_tokens,
    spatial_to_tokens,
    ramp_linear,
    teacher_forward_fixed,
    log_spatial_compare_first_sample,
    _flatten_spatial_tokens,
    _cosine_sim,
    _batch_retrieval_top1,
    _batch_retrieval_mrr,
    _centered_kernel_alignment,
    _style_gram_loss,
    _spatial_energy,
    _corrcoef_1d,
    _topk_activation_f1,
    _pearson_corr_per_sample,
    _sobel_mag_2d,
    _laplacian_highpass_depthwise,
    build_param_groups,
)


def _normalize_batch(x: torch.Tensor, mean, std) -> torch.Tensor:
    """Normalize (B,3,H,W) float [0,1] tensor. In-place friendly."""
    m = torch.tensor(list(mean), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(list(std), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return x.sub_(m).div_(s)


def _empty_accumulator():
    """Return a fresh per-source accumulator dict."""
    return {
        "agg": {},
        "n": 0,
        "summary_cos": [],
        "spatial_cos": [],
        "edge_corr": [],
        "scalar_sums": {},
    }


class DistillLightningModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
        model: DistillModel,
        teacher,
        teacher_proc,
        Ct: int,
        Dt: int,
        Ht: int,
        Wt: int,
        steps_per_epoch: int = 1,
    ):
        super().__init__()
        self.automatic_optimization = False  # manual opt to match old loop memory profile
        self.cfg = cfg
        self.model = model
        # Store teacher WITHOUT registering as a submodule —
        # prevents Lightning from calling .train() on it (must stay eval)
        # and from including its params in state_dict / device management.
        object.__setattr__(self, 'teacher', teacher)
        object.__setattr__(self, 'teacher_proc', teacher_proc)
        self.Ct = Ct
        self.Dt = Dt
        self.Ht = Ht
        self.Wt = Wt

        self.loss_w = LossWeights(
            lambda_summary=cfg.loss.lambda_summary,
            lambda_spatial=cfg.loss.lambda_spatial,
            lambda_mse=cfg.loss.lambda_mse,
        )

        # Mutable loss_state dict for hotcb live control
        self.loss_state = {
            "weights": {
                "lambda_summary": cfg.loss.lambda_summary,
                "lambda_spatial": cfg.loss.lambda_spatial,
                "lambda_mse": cfg.loss.lambda_mse,
                "ramp_mse_sp_end": cfg.loss.ramp.mse_sp.end,
                "ramp_grad_end": cfg.loss.ramp.grad.end,
                "ramp_cons_summary_end": cfg.loss.ramp.cons_summary.end,
                "ramp_cons_spatial_end": cfg.loss.ramp.cons_spatial.end,
                "aug_strength_end": cfg.augmentation.strength_end,
            },
            "terms": {
                "lambda_summary": True,
                "lambda_spatial": True,
                "lambda_mse": True,
                "ramp_mse_sp_end": True,
                "ramp_grad_end": True,
                "ramp_cons_summary_end": True,
                "ramp_cons_spatial_end": True,
                "aug_strength_end": True,
            },
        }

        self.best_val_loss = None

        self.checkpoint_dir = Path(cfg.experiment.root) / "checkpoints"

        # Build augmentation pipeline once
        self.aug_pipeline = build_augmentation_pipeline(cfg.augmentation)

        # Manual grad accumulation & AMP
        self._grad_accum = cfg.training.grad_accum
        self._scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.amp)
        self._micro_step = 0

        # Per-source val accumulators (initialized by set_val_source_names)
        self._val_source_names: list[str] = []
        self._val_per_source: dict[str, dict] = {}
        self._last_alignment_score = 0.0

        self._steps_per_epoch = steps_per_epoch
        self._cached_opt = None

    def set_val_source_names(self, names: list[str]):
        """Called after dm.setup() to configure per-source accumulators."""
        self._val_source_names = list(names)
        self._reset_val_accumulators()

    def _reset_val_accumulators(self):
        self._val_per_source = {src: _empty_accumulator() for src in self._val_source_names}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img_u8, pil_rs, keys = batch
        device = self.device
        if self._cached_opt is None:
            opt_result = self.optimizers()
            # Normalize to list for uniform handling (single AdamW or [AdamW, Muon])
            if isinstance(opt_result, (list, tuple)):
                self._cached_opt = list(opt_result)
            else:
                self._cached_opt = [opt_result]
        opts = self._cached_opt

        # 1. Teacher forward (no grad)
        t_summary, t_spatial_tokens = teacher_forward_fixed(
            teacher=self.teacher,
            proc=self.teacher_proc,
            pil_imgs=pil_rs,
            device=str(device),
            size=self.cfg.data.size,
            amp=self.cfg.teacher.amp,
        )
        del pil_rs  # free PIL list
        # Flush cached blocks from teacher's huge attention intermediates
        # so student can allocate contiguous memory for forward+backward
        torch.cuda.empty_cache()

        # 2. Ramp loss weights (warmup_frac is fraction of first epoch)
        #    Read live-adjustable endpoints from loss_state (hotcb can modify these)
        ws = self.loss_state["weights"]
        self.loss_w.lambda_summary = ws["lambda_summary"]
        self.loss_w.lambda_spatial = ws["lambda_spatial"]
        self.loss_w.lambda_mse = ws["lambda_mse"]

        spe = self._steps_per_epoch
        self.loss_w.lambda_sp_mse = ramp_linear(
            step=self.global_step,
            warmup_steps=int(self.cfg.loss.ramp.mse_sp.warmup_frac * spe),
            start=self.cfg.loss.ramp.mse_sp.start,
            end=ws["ramp_mse_sp_end"],
        )
        self.loss_w.lambda_grad = ramp_linear(
            step=self.global_step,
            warmup_steps=int(self.cfg.loss.ramp.grad.warmup_frac * spe),
            start=self.cfg.loss.ramp.grad.start,
            end=ws["ramp_grad_end"],
        )

        # 3. Prepare clean and augmented inputs on CPU
        aug_strength = ramp_linear(
            self.global_step,
            int(self.cfg.augmentation.warmup_frac * spe),
            self.cfg.augmentation.strength_start,
            ws["aug_strength_end"],
        )

        # Periodically log image grids (before we consume img_u8)
        log_images = (
            self.global_step % (self.cfg.logging.log_every * self.cfg.logging.image_log_multiplier) == 0
            and self.global_step > 0
        )

        # Clean input: float/255 + normalize (no augmentation)
        x_clean_float = img_u8.float().div_(255.0)  # CPU: (B,3,H,W) float [0,1]

        # Augmented input (alpha-blended by strength)
        x_aug_float = apply_student_augmentations(img_u8, aug_strength, self.aug_pipeline)  # CPU: (B,3,H,W) float [0,1]

        if log_images:
            self._log_image_grids(img_u8, x_aug_float, "train")

        del img_u8  # free CPU uint8 batch

        x_clean = _normalize_batch(x_clean_float, self.cfg.data.normalize.mean, self.cfg.data.normalize.std)
        x_clean = x_clean.to(device, non_blocking=True)
        del x_clean_float

        x_aug = _normalize_batch(x_aug_float, self.cfg.data.normalize.mean, self.cfg.data.normalize.std)
        x_aug = x_aug.to(device, non_blocking=True)
        del x_aug_float

        # 4. Dual student forward + losses under autocast
        t_summary = t_summary.to(device, non_blocking=True).float()
        t_spatial_tokens = t_spatial_tokens.to(device, non_blocking=True).float()

        lambda_cs = ramp_linear(
            step=self.global_step,
            warmup_steps=int(self.cfg.loss.ramp.cons_summary.warmup_frac * spe),
            start=self.cfg.loss.ramp.cons_summary.start,
            end=ws["ramp_cons_summary_end"],
        )
        lambda_csp = ramp_linear(
            step=self.global_step,
            warmup_steps=int(self.cfg.loss.ramp.cons_spatial.warmup_frac * spe),
            start=self.cfg.loss.ramp.cons_spatial.start,
            end=ws["ramp_cons_spatial_end"],
        )

        # --- Phase A: clean forward → distill loss → backward (frees clean graph) ---
        with torch.amp.autocast("cuda", enabled=self.cfg.training.amp):
            s_sum_c, s_tokens_c, s_sp_c, (f2, f3) = self.model(x_clean)
            del x_clean, f2, f3

            out = compute_losses(
                s_sum_c, s_tokens_c, s_sp_c, t_summary, t_spatial_tokens,
                self.loss_w, self.cfg.loss.grad_eps, self.Ht, self.Wt,
            )
            distill_loss = out["loss"] / self._grad_accum

        log_vals = {k: v.item() for k, v in out.items()}

        # Detach clean outputs as consistency targets BEFORE backward frees the graph
        s_sum_c_det = s_sum_c.detach()
        s_tokens_c_det = s_tokens_c.detach()

        self._scaler.scale(distill_loss).backward()
        del distill_loss, out, s_sum_c, s_tokens_c, s_sp_c

        # --- Phase B: augmented forward → consistency loss → backward ---
        with torch.amp.autocast("cuda", enabled=self.cfg.training.amp):
            s_sum_a, s_tokens_a, s_sp_a, _ = self.model(x_aug)
            del x_aug, s_sp_a

            cons_summary = cosine_loss(s_sum_a, s_sum_c_det)
            cons_spatial = cosine_loss_spatial_tokens(s_tokens_a, s_tokens_c_det)
            loss_consistency = (lambda_cs * cons_summary + lambda_csp * cons_spatial) / self._grad_accum

        log_vals["cons_summary"] = cons_summary.item()
        log_vals["cons_spatial"] = cons_spatial.item()
        log_vals["loss_cons"] = (lambda_cs * cons_summary.item() + lambda_csp * cons_spatial.item())
        log_vals["w/mse_sp"] = self.loss_w.lambda_sp_mse
        log_vals["w/grad"] = self.loss_w.lambda_grad
        log_vals["w/cons_summary"] = lambda_cs
        log_vals["w/cons_spatial"] = lambda_csp
        log_vals["aug_strength"] = aug_strength

        self._scaler.scale(loss_consistency).backward()
        del loss_consistency, cons_summary, cons_spatial
        del s_sum_a, s_tokens_a, s_sum_c_det, s_tokens_c_det
        del t_summary, t_spatial_tokens

        self._micro_step += 1

        # 8. Optimizer step every grad_accum micro-steps
        if self._micro_step >= self._grad_accum:
            for opt in opts:
                self._scaler.unscale_(opt)
            # Compute grad norm before clipping (after unscale, before step)
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.float().norm().item() ** 2
            log_vals["grad_norm"] = grad_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
            for opt in opts:
                self._scaler.step(opt)
            self._scaler.update()
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            self._micro_step = 0

        # Log optimizer state
        log_vals["lr"] = opts[0].param_groups[0]["lr"]
        log_vals["weight_decay"] = opts[0].param_groups[0].get("weight_decay", 0.0)

        # 10. Logging — write directly to TB, bypass Lightning's metric accumulation
        tb = self.logger.experiment
        step = self.global_step
        for k, v in log_vals.items():
            tb.add_scalar(f"train/{k}", v, step)
            self.log(f"train_{k}", v)

        # Memory tracking
        if step % self.cfg.logging.mem_track_interval == 0 and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            resv = torch.cuda.memory_reserved() / 1024**2
            print(f"[MEM] step={step} allocated={alloc:.0f}MB reserved={resv:.0f}MB")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        src = self._val_source_names[dataloader_idx] if self._val_source_names else "val"
        acc = self._val_per_source.get(src)
        if acc is None:
            acc = _empty_accumulator()
            self._val_per_source[src] = acc

        img_u8, pil_rs, keys = batch
        device = self.device

        # Teacher forward
        t_summary, t_spatial_tokens = teacher_forward_fixed(
            teacher=self.teacher,
            proc=self.teacher_proc,
            pil_imgs=pil_rs,
            device=str(device),
            size=self.cfg.data.size,
            amp=self.cfg.teacher.amp,
        )
        del pil_rs

        # Student: no augmentations, normalize on CPU then move to GPU
        x = img_u8.float().div_(255.0)
        del img_u8
        _normalize_batch(x, self.cfg.data.normalize.mean, self.cfg.data.normalize.std)
        x = x.to(device, non_blocking=True)

        t_summary = t_summary.to(device, non_blocking=True).float()
        t_spatial_tokens = t_spatial_tokens.to(device, non_blocking=True).float()

        with torch.amp.autocast("cuda", enabled=self.cfg.training.amp):
            s_sum, s_tokens, s_sp, _ = self.model(x)
            out = compute_losses(
                s_sum, s_tokens, s_sp, t_summary, t_spatial_tokens,
                self.loss_w, self.cfg.loss.grad_eps, self.Ht, self.Wt,
            )

        # Accumulate losses as floats
        if not acc["agg"]:
            acc["agg"] = {k: 0.0 for k in out}
        for k in out:
            acc["agg"][k] += out[k].item()
        del out

        # Spatial compare logs for first few batches
        vsc = self.cfg.logging.val_spatial_compare
        if batch_idx % vsc.batch_mod == 0 and batch_idx < vsc.max_batches:
            tb = self.logger.experiment
            log_spatial_compare_first_sample(
                tb_writer=tb,
                step=self.global_step,
                s_spatial_bdhw=s_sp,
                t_tokens_btd=t_spatial_tokens,
                Ht=self.Ht,
                Wt=self.Wt,
                tag_prefix=f"eval/{src}/spatial_compare_{batch_idx}_first",
                max_side=vsc.max_side,
            )

        # Move everything to CPU for diagnostics, free GPU immediately
        s_sum_c = s_sum.float().cpu()
        t_summary_c = t_summary.float().cpu()
        s_sp_c = s_sp.float().cpu()
        t_spatial_c = t_spatial_tokens.float().cpu()
        x_c = x.float().cpu()
        del s_sum, t_summary, s_tokens, s_sp, t_spatial_tokens, x
        torch.cuda.empty_cache()

        acc["n"] += 1

        # All diagnostics on CPU
        s_tokens2 = _flatten_spatial_tokens(s_sp_c)        # (B,T,D)
        t_tokens2 = t_spatial_c                              # already (B,T,D)
        B, Tt, Dt_ = t_tokens2.shape
        t_map = t_tokens2.transpose(1, 2).contiguous().reshape(B, Dt_, self.Ht, self.Wt)

        acc["summary_cos"].append(_cosine_sim(s_sum_c, t_summary_c, dim=-1))
        acc["spatial_cos"].append(_cosine_sim(s_tokens2, t_tokens2, dim=-1).flatten())

        sums = acc["scalar_sums"]
        sums["retrieval_top1"] = sums.get("retrieval_top1", 0.0) + float(_batch_retrieval_top1(s_sum_c, t_summary_c))
        sums["retrieval_mrr"] = sums.get("retrieval_mrr", 0.0) + float(_batch_retrieval_mrr(s_sum_c, t_summary_c))
        sums["summary_cka"] = sums.get("summary_cka", 0.0) + float(_centered_kernel_alignment(s_sum_c, t_summary_c))

        sums["summary_mse_mean"] = sums.get("summary_mse_mean", 0.0) + float(((s_sum_c - t_summary_c) ** 2).mean())
        sums["summary_norm_ratio"] = sums.get("summary_norm_ratio", 0.0) + float((s_sum_c.norm(dim=-1) / (t_summary_c.norm(dim=-1) + 1e-8)).mean())
        sums["summary_mean_abs_diff"] = sums.get("summary_mean_abs_diff", 0.0) + float((s_sum_c.mean(dim=0) - t_summary_c.mean(dim=0)).abs().mean())
        sums["summary_std_abs_diff"] = sums.get("summary_std_abs_diff", 0.0) + float((s_sum_c.std(dim=0) - t_summary_c.std(dim=0)).abs().mean())
        del s_sum_c, t_summary_c

        sums["spatial_mse_mean"] = sums.get("spatial_mse_mean", 0.0) + float(((s_tokens2 - t_tokens2) ** 2).mean())
        sums["spatial_norm_ratio"] = sums.get("spatial_norm_ratio", 0.0) + float((s_tokens2.norm(dim=-1) / (t_tokens2.norm(dim=-1) + 1e-8)).mean())
        sums["spatial_style_gram"] = sums.get("spatial_style_gram", 0.0) + float(_style_gram_loss(s_tokens2, t_tokens2))

        s_energy = _spatial_energy(s_sp_c)
        t_energy = _spatial_energy(t_map)
        sums["spatial_energy_ratio"] = sums.get("spatial_energy_ratio", 0.0) + float((s_energy / (t_energy + 1e-8)).mean())

        s_meanD = s_tokens2.mean(dim=1)
        t_meanD = t_tokens2.mean(dim=1)
        sums["spatial_meanD_corr"] = sums.get("spatial_meanD_corr", 0.0) + float(_corrcoef_1d(s_meanD.flatten(), t_meanD.flatten()))
        sums["act_f1"] = sums.get("act_f1", 0.0) + float(_topk_activation_f1(s_tokens2, t_tokens2))
        del s_tokens2, t_tokens2, s_meanD, t_meanD

        # Edge alignment (all CPU)
        x_gray = x_c.mean(dim=1, keepdim=True)
        g_img = _sobel_mag_2d(x_gray)
        del x_c, x_gray
        s_energy_map = torch.sqrt((s_sp_c * s_sp_c).sum(dim=1, keepdim=True) + 1e-8)
        g_feat = _sobel_mag_2d(s_energy_map)
        del s_energy_map
        g_img_ds = F.interpolate(g_img, size=(self.Ht, self.Wt), mode="bilinear", align_corners=False)
        del g_img
        acc["edge_corr"].append(_pearson_corr_per_sample(g_feat.flatten(1), g_img_ds.flatten(1)))
        del g_feat, g_img_ds

        # HF metrics (all CPU)
        s_hf = _laplacian_highpass_depthwise(s_sp_c)
        t_hf = _laplacian_highpass_depthwise(t_map)
        del s_sp_c, t_map, t_spatial_c
        s_hf_tok = _flatten_spatial_tokens(s_hf)
        t_hf_tok = _flatten_spatial_tokens(t_hf)
        sums["hf_cos"] = sums.get("hf_cos", 0.0) + float(_cosine_sim(s_hf_tok, t_hf_tok, dim=-1).mean())
        sums["hf_mse"] = sums.get("hf_mse", 0.0) + float(F.mse_loss(s_hf, t_hf))
        s_hf_energy = _spatial_energy(s_hf)
        t_hf_energy = _spatial_energy(t_hf)
        sums["hf_energy_ratio"] = sums.get("hf_energy_ratio", 0.0) + float((s_hf_energy / (t_hf_energy + 1e-8)).mean())
        del s_hf, t_hf, s_hf_tok, t_hf_tok

    def on_validation_epoch_end(self):
        # Just defragment GPU — reporting happens in on_train_epoch_end
        torch.cuda.empty_cache()

    def _compute_source_metrics(self, src: str, acc: dict) -> dict:
        """Compute and log all metrics for one val source. Returns key values."""
        n = acc["n"]
        if n == 0:
            return {}

        tb = self.logger.experiment
        step = self.global_step
        quantiles = list(self.cfg.logging.quantiles)

        # Mean losses
        val_loss = acc["agg"].get("loss", 0.0) / n
        for k, v in acc["agg"].items():
            tb.add_scalar(f"val/{src}/{k}", v / n, step)

        # Quantile metrics
        if acc["summary_cos"]:
            cat = torch.cat(acc["summary_cos"], dim=0)
            tb.add_scalar(f"val/{src}/summary_cos_mean", cat.mean().item(), step)
            for q in quantiles:
                tb.add_scalar(f"val/{src}/summary_cos_p{int(q*100):02d}", torch.quantile(cat, q).item(), step)

        if acc["spatial_cos"]:
            cat = torch.cat(acc["spatial_cos"], dim=0)
            tb.add_scalar(f"val/{src}/spatial_cos_mean", cat.mean().item(), step)
            for q in quantiles:
                tb.add_scalar(f"val/{src}/spatial_cos_p{int(q*100):02d}", torch.quantile(cat, q).item(), step)

        if acc["edge_corr"]:
            cat = torch.cat(acc["edge_corr"], dim=0)
            tb.add_scalar(f"val/{src}/edge_align_corr_mean", cat.mean().item(), step)
            for q in quantiles:
                tb.add_scalar(f"val/{src}/edge_align_corr_p{int(q*100):02d}", torch.quantile(cat, q).item(), step)

        # Scalar means
        for k, v in acc["scalar_sums"].items():
            tb.add_scalar(f"val/{src}/{k}", v / n, step)

        # Composite alignment score
        cos_sum_sim = 1.0 - acc["agg"].get("cos_sum", 0.0) / n
        cos_sp_sim  = 1.0 - acc["agg"].get("cos_sp", 0.0) / n
        hf_cos_mean = acc["scalar_sums"].get("hf_cos", 0.0) / n
        hf_mse_mean = acc["scalar_sums"].get("hf_mse", 0.0) / n
        act_f1_mean = acc["scalar_sums"].get("act_f1", 0.0) / n

        alignment_score = (
            0.30 * cos_sum_sim + 0.30 * cos_sp_sim
            + 0.20 * hf_cos_mean - 0.10 * hf_mse_mean
            + 0.30 * act_f1_mean
        )
        tb.add_scalar(f"val/{src}/act_f1", act_f1_mean, step)
        tb.add_scalar(f"val/{src}/alignment_score", alignment_score, step)

        return {"alignment_score": alignment_score, "val_loss": val_loss}

    def on_train_epoch_end(self):
        # Check if any source has data
        total_n = sum(acc["n"] for acc in self._val_per_source.values())
        if total_n == 0:
            return

        tb = self.logger.experiment
        step = self.global_step

        # Compute per-source metrics
        source_results = {}
        for src in self._val_source_names:
            acc = self._val_per_source.get(src)
            if acc and acc["n"] > 0:
                source_results[src] = self._compute_source_metrics(src, acc)

        # Group-level alignment scores
        oi_score = source_results.get("oi_val", {}).get("alignment_score", 0.0)
        off_dist_sources = [name for name in self._val_source_names if name != "oi_val"]
        off_dist_scores = [source_results[s]["alignment_score"] for s in off_dist_sources if s in source_results]

        if off_dist_scores:
            off_dist_avg = sum(off_dist_scores) / len(off_dist_scores)
        else:
            off_dist_avg = 0.0

        tb.add_scalar("val/in_dist/alignment_score", oi_score, step)
        tb.add_scalar("val/off_dist/alignment_score", off_dist_avg, step)

        # Overall: 0.25 × oi_val + 0.375 × coco_train + 0.375 × imagenet_train
        # Generalized: 0.25 × oi_val + 0.75 spread across off-dist sources
        if off_dist_scores:
            off_weight_each = 0.75 / len(off_dist_scores)
            overall = 0.25 * oi_score + sum(off_weight_each * s for s in off_dist_scores)
        else:
            overall = oi_score

        tb.add_scalar("val/alignment_score", overall, step)
        self._last_alignment_score = overall

        val_metrics = {"alignment_score": overall, "in_dist_alignment": oi_score, "off_dist_alignment": off_dist_avg}
        for src, r in source_results.items():
            for k, v in r.items():
                val_metrics[f"{src}_{k}"] = v
        for k, v in val_metrics.items():
            self.log(f"val_{k}", v)

        # Checkpointing on val improvement (use oi_val loss as primary)
        oi_loss = source_results.get("oi_val", {}).get("val_loss")
        if oi_loss is not None:
            if self.best_val_loss is None or oi_loss < self.best_val_loss:
                self.best_val_loss = oi_loss
                self._save_checkpoint(step, oi_loss)

        # Print epoch summary
        parts = []
        for src in self._val_source_names:
            r = source_results.get(src, {})
            parts.append(f"{src}: loss={r.get('val_loss', 0):.4f} align={r.get('alignment_score', 0):.4f}")
        n_batches = {src: self._val_per_source[src]["n"] for src in self._val_source_names}
        print(
            f"[VAL] epoch={self.current_epoch} step={step} overall_align={overall:.4f} "
            f"| {' | '.join(parts)} "
            f"(batches: {n_batches})"
        )

        # Reset accumulators for next epoch
        self._reset_val_accumulators()

    def on_train_epoch_start(self):
        # Update dataset seed for pad-vs-squash variation
        dl = self.trainer.train_dataloader
        if dl is not None and hasattr(dl, 'dataset'):
            ds = dl.dataset
            if hasattr(ds, 'set_epoch'):
                ds.set_epoch(self.current_epoch)

    def configure_optimizers(self):
        opt_type = getattr(self.cfg.training, "optimizer_type", "adamw")

        if opt_type == "adamw_muon":
            # Combined AdamW (norms/biases) + Muon (weight matrices)
            # Collect all submodules into one namespace for param splitting
            from itertools import chain
            all_named = list(chain(
                self.model.student.named_parameters(),
                self.model.sum_head.named_parameters(),
                self.model.sp_head.named_parameters(),
            ))
            # Build a temporary nn.Module-like wrapper so build_adamw_muon_optimizer works
            combined = torch.nn.Module()
            for i, (name, p) in enumerate(all_named):
                combined.register_parameter(f"p{i}_{name.replace('.', '_')}", p)

            muon_cfg = self.cfg.training.muon
            optimizers = build_adamw_muon_optimizer(
                model=combined,
                lr_adamw=self.cfg.training.lr,
                lr_muon=muon_cfg.lr,
                wd=self.cfg.training.wd,
                betas=tuple(self.cfg.training.optimizer.betas),
                eps=self.cfg.training.optimizer.eps,
                muon_momentum=muon_cfg.momentum,
                muon_nesterov=muon_cfg.nesterov,
                muon_ns_steps=muon_cfg.ns_steps,
            )
            return optimizers
        else:
            # Standard AdamW
            params = []
            params += build_param_groups(self.model.student, self.cfg.training.wd)
            params += build_param_groups(self.model.sum_head, self.cfg.training.wd)
            params += build_param_groups(self.model.sp_head, self.cfg.training.wd)
            betas = tuple(self.cfg.training.optimizer.betas)
            opt = torch.optim.AdamW(
                params, lr=self.cfg.training.lr,
                betas=betas, eps=self.cfg.training.optimizer.eps,
            )
            return opt

    def _save_checkpoint(self, step: int, val_loss: float):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.checkpoint_dir / f"epoch_{self.current_epoch}_step_{step}_val_loss_{val_loss:.3f}.pth"
        torch.save(
            {
                "student_name": self.cfg.model.student_variant,
                "size": self.cfg.data.size,
                "patch_size": self.cfg.data.patch_size,
                "teacher_summary_dim": self.model.sum_head.fc.out_features,
                "teacher_spatial_dim": self.model.sp_head.conv.out_channels,
                "student_state_dict": self.model.student.state_dict(),
                "sum_head_state_dict": self.model.sum_head.state_dict(),
                "sp_head_state_dict": self.model.sp_head.state_dict(),
            },
            save_path,
        )
        print(f"Saved checkpoint: {save_path}")

    @torch.no_grad()
    def _log_image_grids(self, img_u8: torch.Tensor, x_augmented: torch.Tensor, prefix: str):
        """Log teacher (un-augmented) and student (augmented) input grids. Both on CPU."""
        tb = self.logger.experiment
        step = self.global_step
        ig = self.cfg.logging.image_grid
        n = min(ig.n, img_u8.shape[0])

        # Teacher input: original uint8 -> [0,1]
        teacher_imgs = img_u8[:n].float().div(255.0)
        teacher_grid = make_grid(teacher_imgs, nrow=ig.nrow, normalize=False)
        tb.add_image(f"{prefix}/teacher_input", teacher_grid, step)

        # Student input: augmented float [0,1]
        student_imgs = x_augmented[:n].detach()
        student_grid = make_grid(student_imgs, nrow=ig.nrow, normalize=False)
        tb.add_image(f"{prefix}/student_input", student_grid, step)
