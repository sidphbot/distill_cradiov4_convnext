import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import timm
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoModel, CLIPImageProcessor

from distill.data import DistillDataModule
from distill.model import (
    DistillModel, SummaryHead, SpatialHead,
    build_param_groups, teacher_forward_fixed,
)
from distill.lightning_module import DistillLightningModule


# ============================================================
# CONFIG
# ============================================================

@dataclass
class TrainConfig:
    student_variant: str = "small"  # tiny|small
    mode: str = "debug"  # debug|full
    size: int = 416
    patch_size: int = 16

    data_list: str = ""          # path to txt file of image paths
    val_frac: float = 0.02       # deterministic split

    # teacher
    teacher_id: str = "nvidia/C-RADIOv4-H"
    teacher_amp: bool = True

    batch_size: int = 36
    num_workers: int = 8
    persistent_workers: bool = True
    prefetch_factor: int = 8

    epochs: int = 1
    lr: float = 1.5e-4
    wd: float = 0.08
    amp: bool = True
    grad_accum: int = 4

    log_every: int = 50
    eval_every: int = 100
    eval_batches: int = 50

    exp_root: Path = Path("/home/burplord/experiments/")
    exp_suffix: str = ""
    seed: int = 42
    lambda_summary: float = 1.0
    lambda_spatial: float = 1.0
    lambda_mse: float = 0.05

    grad_checkpointing: bool = False

    mse_sp_w_start: float = 0.5
    mse_sp_w_end: float = 1.5
    mse_sp_w_warmup_frac: float = 0.2
    grad_w_start: float = 0.1
    grad_w_end: float = 0.3
    grad_w_warmup_frac: float = 0.35

    grad_eps: float = 1e-3

    # Data caps (0 = no cap)
    train_cap: int = 0
    val_cap: int = 0


def pick_student_name(variant: str) -> str:
    if variant == "tiny":
        return "convnext_tiny.dinov3_lvd1689m"
    if variant == "small":
        return "convnext_small.dinov3_lvd1689m"
    raise ValueError(f"Unknown student_variant={variant}")


def build_model(
    cfg: TrainConfig,
    device: str,
    Ct: int,
    Dt: int,
    Ht: int,
    Wt: int,
    student_name: str,
) -> DistillModel:
    student = timm.create_model(
        student_name,
        pretrained=True,
        features_only=True,
        out_indices=(2, 3),
    ).to(device)

    if cfg.grad_checkpointing:
        if hasattr(student, "set_grad_checkpointing"):
            student.set_grad_checkpointing(True)
        elif hasattr(student, "grad_checkpointing"):
            student.grad_checkpointing = True

    # infer student dims
    with torch.no_grad():
        f2, f3 = student(torch.randn(1, 3, cfg.size, cfg.size, device=device))
        Cs_sp = f2.shape[1]
        Cs = f3.shape[1]

    sum_head = SummaryHead(Cs, Ct).to(device)
    sp_head = SpatialHead(Cs_sp, Dt).to(device)

    model = DistillModel(student=student, sum_head=sum_head, sp_head=sp_head, Ht=Ht, Wt=Wt).to(device)
    return model


# ============================================================
# CLI
# ============================================================

def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_variant", type=str, default="small", choices=["tiny", "small"])
    ap.add_argument("--mode", type=str, default="debug", choices=["debug", "full"])
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--patch_size", type=int, default=16)

    ap.add_argument("--data_list", type=str, default="", help="Text file with one image path per line.")
    ap.add_argument("--image_dir", type=str, default="", help="Directory of images; auto-generates data_list in cwd.")
    ap.add_argument("--val_frac", type=float, default=0.02)

    ap.add_argument("--teacher_id", type=str, default="nvidia/C-RADIOv4-H")
    ap.add_argument("--teacher_amp", action="store_true", default=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    ap.set_defaults(persistent_workers=True)
    ap.add_argument("--prefetch_factor", type=int, default=8)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--wd", type=float, default=0.08)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=50)

    ap.add_argument("--exp_root", type=Path, default=Path("/home/burplord/experiments/"))
    ap.add_argument("--exp_suffix", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lambda_summary", type=float, default=1.0)
    ap.add_argument("--lambda_spatial", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=0.25)

    ap.add_argument("--grad_checkpointing", action="store_true")
    ap.add_argument("--mse_sp_w_start", type=float, default=0.5)
    ap.add_argument("--mse_sp_w_end", type=float, default=1.5)
    ap.add_argument("--mse_sp_w_warmup_frac", type=float, default=0.2)
    ap.add_argument("--grad_w_start", type=float, default=0.1)
    ap.add_argument("--grad_w_end", type=float, default=1.3)
    ap.add_argument("--grad_w_warmup_frac", type=float, default=0.35)
    ap.add_argument("--grad_eps", type=float, default=1e-3)

    a = ap.parse_args()

    assert not (a.data_list and a.image_dir), \
        "Provide --data_list or --image_dir, not both."

    if a.image_dir and not a.data_list:
        a.data_list = _generate_data_list(a.image_dir)

    # Debug mode overrides
    if a.mode == "debug":
        a.eval_every = 5
        a.eval_batches = 5
        a.train_cap = 5000
        a.val_cap = 1000
        a.exp_suffix = "debug" if not a.exp_suffix else a.exp_suffix + "_debug"

    d = vars(a)
    d.pop("image_dir", None)
    return TrainConfig(**d)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _generate_data_list(image_dir: str) -> str:
    """Walk image_dir, write paths to ./data_list.txt, return its path."""
    import os
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


def main():
    cfg = parse_args()

    if not cfg.data_list:
        raise SystemExit("--data_list or --image_dir is required")

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Build teacher
    teacher_proc = CLIPImageProcessor.from_pretrained(cfg.teacher_id)
    teacher = AutoModel.from_pretrained(cfg.teacher_id, trust_remote_code=True).to(device).eval().half()
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
        size=cfg.size,
        amp=cfg.teacher_amp,
    )
    Ct = int(t_sum0.shape[-1])
    Dt = int(t_tok0.shape[-1])
    Ht = cfg.size // cfg.patch_size
    Wt = cfg.size // cfg.patch_size
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
    student_name = pick_student_name(cfg.student_variant)
    model = build_model(cfg, device, Ct, Dt, Ht, Wt, student_name)

    # 5. Create Lightning module
    exp_name = f"exp_{time.strftime('%Y%m%d%H%M')}_{cfg.exp_suffix}" if cfg.exp_suffix else f"exp_{time.strftime('%Y%m%d%H%M')}"
    lit_module = DistillLightningModule(
        cfg=cfg,
        model=model,
        teacher=teacher,
        teacher_proc=teacher_proc,
        Ct=Ct, Dt=Dt, Ht=Ht, Wt=Wt,
    )
    lit_module.checkpoint_dir = cfg.exp_root / exp_name / "checkpoints"

    # 6. Logger & Trainer
    tb_logger = TensorBoardLogger(
        save_dir=str(cfg.exp_root / exp_name),
        name="tb",
    )

    # val_check_interval counts raw batches, but eval_every is in optimizer steps.
    # Multiply by grad_accum to match the old loop's step-based eval schedule.
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=tb_logger,
        val_check_interval=cfg.eval_every * cfg.grad_accum,
        limit_val_batches=cfg.eval_batches,
        log_every_n_steps=cfg.log_every,
        enable_progress_bar=True,
    )

    # 7. Train
    trainer.fit(lit_module, datamodule=dm)


if __name__ == "__main__":
    main()
