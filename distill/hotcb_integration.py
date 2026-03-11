"""hotcb integration for distillation training.

Uses the Lightning adapter (Option C) from hotcb to expose all tunable knobs
and emit comprehensive metrics for live training optimization.

Provides:
  - build_hotcb_callback()  — for use inside the normal launcher
  - train()                 — hotcb-compatible training function for `hotcb launch`

Launch via hotcb CLI:
    hotcb launch --train-fn distill.hotcb_integration:train \
        --autopilot ai_suggest --key-metric alignment_score

Or attach to an already-running training:
    hotcb serve --dir /path/to/experiment --autopilot ai_suggest --key-metric alignment_score
"""

import math
import os
import threading
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModel, CLIPImageProcessor

from distill.data import DistillDataModule
from distill.model import teacher_forward_fixed
from distill.launcher import build_model, build_hotcb_callback
from distill.lightning_module import DistillLightningModule


class _StopEventCallback(pl.Callback):
    """Checks stop_event each training step and stops the trainer cleanly."""

    def __init__(self, stop_event: threading.Event):
        super().__init__()
        self._stop_event = stop_event

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._stop_event.is_set():
            trainer.should_stop = True


def train(run_dir: str, max_steps: int, step_delay: float,
          stop_event: threading.Event):
    """hotcb-compatible training function.

    Follows the hotcb training function contract so this can be launched via:
        hotcb launch --train-fn distill.hotcb_integration:train \\
            --autopilot ai_suggest --key-metric alignment_score

    Uses run_dir for all hotcb JSONL I/O. Config is loaded from the normal
    distill config.yaml (with CLI overrides via DISTILL_CONFIG_OVERRIDES env var).
    """
    # ── Load config ──────────────────────────────────────────────────
    config_path = os.environ.get("DISTILL_CONFIG", "distill/config.yaml")
    cfg = OmegaConf.load(config_path)

    # Allow dotlist overrides via env var (space-separated)
    overrides_str = os.environ.get("DISTILL_CONFIG_OVERRIDES", "")
    if overrides_str.strip():
        cli_cfg = OmegaConf.from_dotlist(overrides_str.strip().split())
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Apply debug overlay
    if cfg.mode == "debug":
        if "debug" in cfg:
            cfg = OmegaConf.merge(cfg, cfg.debug)

    # Force hotcb enabled — we're being launched by hotcb
    if not hasattr(cfg, "hotcb"):
        cfg.hotcb = OmegaConf.create({"enabled": True, "key_metric": "alignment_score", "debounce_steps": 10})
    else:
        cfg.hotcb.enabled = True

    if not cfg.data.image_dir:
        raise RuntimeError("data.image_dir is required — set via DISTILL_CONFIG_OVERRIDES "
                           "or in your config.yaml")

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[hotcb train] Device: {device}")

    # ── Build teacher ────────────────────────────────────────────────
    teacher_proc = CLIPImageProcessor.from_pretrained(cfg.teacher.id)
    teacher = (AutoModel.from_pretrained(cfg.teacher.id, trust_remote_code=True)
               .to(device).eval().half())
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Data ─────────────────────────────────────────────────────────
    dm = DistillDataModule(cfg)
    dm.setup()

    # ── Infer teacher dims ───────────────────────────────────────────
    sample_batch = next(iter(dm.train_dataloader()))
    _, pil_rs0, _ = sample_batch
    t_sum0, t_tok0 = teacher_forward_fixed(
        teacher=teacher, proc=teacher_proc, pil_imgs=pil_rs0,
        device=device, size=cfg.data.size, amp=cfg.teacher.amp,
    )
    Ct, Dt = int(t_sum0.shape[-1]), int(t_tok0.shape[-1])
    Ht = Wt = cfg.data.size // cfg.data.patch_size
    del t_sum0, t_tok0, pil_rs0, sample_batch
    torch.cuda.empty_cache()

    # ── Build student model ──────────────────────────────────────────
    student_name = cfg.model.student_models[cfg.model.student_variant]
    model = build_model(cfg, device, Ct, Dt, Ht, Wt, student_name)

    steps_per_epoch = math.ceil(len(dm.train_ds) / cfg.dataloader.batch_size)

    # ── Lightning module ─────────────────────────────────────────────
    lit_module = DistillLightningModule(
        cfg=cfg, model=model, teacher=teacher, teacher_proc=teacher_proc,
        Ct=Ct, Dt=Dt, Ht=Ht, Wt=Wt, steps_per_epoch=steps_per_epoch,
    )
    lit_module.checkpoint_dir = Path(run_dir) / "checkpoints"
    lit_module.set_val_source_names(dm.val_source_names)

    # ── hotcb callback + metrics collector ───────────────────────────
    hotcb_cb = build_hotcb_callback(cfg, run_dir=run_dir)
    callbacks = []
    if hotcb_cb is not None:
        callbacks.append(hotcb_cb)

    # ── stop_event callback ──────────────────────────────────────────
    callbacks.append(_StopEventCallback(stop_event))

    # ── Logger & Trainer ─────────────────────────────────────────────
    tb_logger = TensorBoardLogger(save_dir=run_dir, name="tb")

    # Compute max_epochs from max_steps
    effective_steps_per_epoch = max(1, steps_per_epoch // cfg.training.grad_accum)
    max_epochs = max(1, math.ceil(max_steps / effective_steps_per_epoch))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        logger=tb_logger,
        val_check_interval=cfg.logging.eval_every * cfg.training.grad_accum,
        limit_val_batches=cfg.logging.eval_batches,
        log_every_n_steps=cfg.logging.log_every,
        enable_progress_bar=True,
        callbacks=callbacks,
    )

    print(f"[hotcb train] Starting training: max_steps={max_steps}, "
          f"steps_per_epoch={steps_per_epoch}, max_epochs={max_epochs}")

    # ── Train ────────────────────────────────────────────────────────
    trainer.fit(lit_module, datamodule=dm)

    print(f"[hotcb train] Training finished at step {trainer.global_step}, "
          f"alignment_score={lit_module._last_alignment_score:.4f}")
