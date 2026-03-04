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
    build_param_groups, teacher_forward_fixed,
)
from distill.lightning_module import DistillLightningModule


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _generate_data_list(image_dir: str) -> str:
    """Walk image_dir, write paths to ./data_list.txt, return its path."""
    root = os.path.abspath(image_dir)
    paths = sorted(
        os.path.join(dp, f)
        for dp, _, fnames in os.walk(root)
        for f in fnames
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not paths:
        raise SystemExit(f"No images found in {root}")
    out = os.path.join(os.getcwd(), "data_list.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(paths) + "\n")
    print(f"Generated {out} with {len(paths)} images from {root}")
    return out


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

    # Handle image_dir -> data_list generation
    image_dir = cfg.data.get("image_dir", "")
    if image_dir and not cfg.data.data_list:
        cfg.data.data_list = _generate_data_list(image_dir)
        cfg.data.image_dir = ""
    elif image_dir and cfg.data.data_list:
        raise SystemExit("Provide data.data_list or data.image_dir, not both.")

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

    if not cfg.data.data_list:
        raise SystemExit("data.data_list or data.image_dir is required")

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

    # 7. Logger & Trainer
    tb_logger = TensorBoardLogger(
        save_dir=str(Path(cfg.experiment.root) / exp_name),
        name="tb",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=tb_logger,
        val_check_interval=cfg.logging.eval_every * cfg.training.grad_accum,
        limit_val_batches=cfg.logging.eval_batches,
        log_every_n_steps=cfg.logging.log_every,
        enable_progress_bar=True,
    )

    # 8. Train
    trainer.fit(lit_module, datamodule=dm)


if __name__ == "__main__":
    main()
