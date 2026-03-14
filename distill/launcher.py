import argparse
import os
import time
from pathlib import Path

import timm
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoModel, CLIPImageProcessor

from distill.data import DistillDataModule
from distill.model import (
    DistillModel, SummaryHead, SpatialHead,
    teacher_forward_fixed,
)
from distill.lightning_module import DistillLightningModule


def build_hotcb_callback(cfg, run_dir: str, mutable_state=None):
    """Build HotCBLightning callback with MetricsCollector.

    Args:
        cfg: OmegaConf config with hotcb section.
        run_dir: Experiment directory for hotcb JSONL files.
        mutable_state: Optional dict to pass explicitly to HotCBLightning
            (belt-and-suspenders with pl_module.mutable_state auto-detect).

    Returns:
        callback or None if disabled/unavailable.
    """
    if not getattr(cfg, "hotcb", None) or not cfg.hotcb.enabled:
        return None

    try:
        from hotcb.kernel import HotKernel
        from hotcb.adapters.lightning import HotCBLightning
        from hotcb.metrics import MetricsCollector
    except ImportError:
        print("[hotcb] hotcb not installed — skipping integration. "
              "Install with: pip install 'hotcb[dashboard]'")
        return None

    os.makedirs(run_dir, exist_ok=True)

    metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
    mc = MetricsCollector(metrics_path)
    kernel = HotKernel(
        run_dir=run_dir,
        metrics_collector=mc,
        debounce_steps=cfg.hotcb.get("debounce_steps", 10),
    )

    callback = HotCBLightning(kernel, mutable_state=mutable_state)
    key_metric = cfg.hotcb.get("key_metric", "alignment_score")
    print(f"[hotcb] Enabled — run_dir={run_dir}")
    print(f"[hotcb] Launch dashboard: hotcb serve --dir {run_dir}")
    print(f"[hotcb] With autopilot:   hotcb serve --dir {run_dir} "
          f"--autopilot ai_suggest --key-metric {key_metric}")

    return callback



def load_config():
    """
    Load config from YAML, merge CLI dotlist overrides, apply debug overlay.

    Usage:
        python -m distill.launcher --config distill/config.yaml mode=debug dataloader.batch_size=16
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="distill/config.yaml", help="Path to YAML config")
    args, overrides = ap.parse_known_args()

    cfg = OmegaConf.load(args.config)

    # Merge CLI dotlist overrides (e.g. dataloader.batch_size=16)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Apply debug overlay
    if cfg.mode == "debug":
        if "debug" in cfg:
            debug_overlay = cfg.debug
            cfg = OmegaConf.merge(cfg, debug_overlay)
        suffix = cfg.experiment.get("suffix", "")
        cfg.experiment.suffix = f"{suffix}_debug" if suffix else "debug"

    return cfg


def build_model(cfg, device, Ct, Dt, Ht, Wt, student_name):
    student = timm.create_model(
        student_name,
        pretrained=True,
        features_only=True,
        out_indices=(2, 3),
    ).to(device)

    if cfg.model.grad_checkpointing:
        if hasattr(student, "set_grad_checkpointing"):
            student.set_grad_checkpointing(True)
        elif hasattr(student, "grad_checkpointing"):
            student.grad_checkpointing = True

    with torch.no_grad():
        f2, f3 = student(torch.randn(1, 3, cfg.data.size, cfg.data.size, device=device))
        Cs_sp = f2.shape[1]
        Cs = f3.shape[1]

    sum_head = SummaryHead(Cs, Ct).to(device)
    sp_head = SpatialHead(Cs_sp, Dt).to(device)

    model = DistillModel(
        student=student, sum_head=sum_head, sp_head=sp_head,
        Ht=Ht, Wt=Wt, spatial_norm_cap=cfg.model.spatial_norm_cap,
    ).to(device)
    return model


def main():
    cfg = load_config()

    if not cfg.data.image_dir:
        raise SystemExit("data.image_dir is required")
    val_frac = getattr(cfg.data, "val_frac", 0.0)
    if val_frac <= 0:
        raise SystemExit("data.val_frac must be > 0 (fraction of image_dir held out for oi_val)")

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Build teacher
    teacher_proc = CLIPImageProcessor.from_pretrained(cfg.teacher.id)
    teacher = AutoModel.from_pretrained(cfg.teacher.id, trust_remote_code=True).to(device).eval().half()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 2. Build data module and setup to get a sample batch
    dm = DistillDataModule(cfg)
    dm.setup()

    # 3. Infer teacher dims from one forward pass
    print("Inferring teacher dims...")
    sample_batch = next(iter(dm.train_dataloader()))
    _, pil_rs0, _ = sample_batch
    t_sum0, t_tok0 = teacher_forward_fixed(
        teacher=teacher,
        proc=teacher_proc,
        pil_imgs=pil_rs0,
        device=device,
        size=cfg.data.size,
        amp=cfg.teacher.amp,
    )
    Ct = int(t_sum0.shape[-1])
    Dt = int(t_tok0.shape[-1])
    Ht = cfg.data.size // cfg.data.patch_size
    Wt = cfg.data.size // cfg.data.patch_size
    Tt = int(t_tok0.shape[1])
    assert Tt == Ht * Wt, f"T={Tt} != Ht*Wt={Ht * Wt}"
    print(f"Teacher dims: Ct={Ct}, Dt={Dt}, Ht={Ht}, Wt={Wt}")
    del t_sum0, t_tok0, pil_rs0, sample_batch
    torch.cuda.empty_cache()

    if device == "cuda":
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU after teacher load: {alloc:.2f}GB allocated, {resv:.2f}GB reserved, {total:.2f}GB total")

    # 4. Build student model
    student_name = cfg.model.student_models[cfg.model.student_variant]
    model = build_model(cfg, device, Ct, Dt, Ht, Wt, student_name)

    # 5. Compute steps per epoch (micro-batch calls to training_step)
    import math
    steps_per_epoch = math.ceil(len(dm.train_ds) / cfg.dataloader.batch_size)
    print(f"Steps per epoch: {steps_per_epoch} ({len(dm.train_ds)} samples / bs {cfg.dataloader.batch_size})")

    # 6. Create Lightning module
    suffix = cfg.experiment.get("suffix", "")
    exp_name = f"exp_{time.strftime('%Y%m%d%H%M')}_{suffix}" if suffix else f"exp_{time.strftime('%Y%m%d%H%M')}"
    lit_module = DistillLightningModule(
        cfg=cfg,
        model=model,
        teacher=teacher,
        teacher_proc=teacher_proc,
        Ct=Ct, Dt=Dt, Ht=Ht, Wt=Wt,
        steps_per_epoch=steps_per_epoch,
    )
    lit_module.checkpoint_dir = Path(cfg.experiment.root) / exp_name / "checkpoints"
    lit_module.set_val_source_names(dm.val_source_names)

    # 7. Logger & Trainer
    exp_dir = str(Path(cfg.experiment.root) / exp_name)
    tb_logger = TensorBoardLogger(
        save_dir=exp_dir,
        name="tb",
    )

    callbacks = []
    hotcb_cb = build_hotcb_callback(cfg, run_dir=exp_dir, mutable_state=lit_module.mutable_state)
    if hotcb_cb is not None:
        callbacks.append(hotcb_cb)

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=tb_logger,
        val_check_interval=cfg.logging.eval_every * cfg.training.grad_accum,
        limit_val_batches=cfg.logging.eval_batches,
        log_every_n_steps=cfg.logging.log_every,
        enable_progress_bar=True,
        callbacks=callbacks,
    )

    # 8. Train
    trainer.fit(lit_module, datamodule=dm)


if __name__ == "__main__":
    main()
