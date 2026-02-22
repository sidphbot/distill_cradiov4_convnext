import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import timm
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from distill.data import find_shards, TarShardDataset, make_collate
from distill.loop import LossWeights, train_one_epoch
from distill.model import DistillModel, SummaryHead, SpatialHead, build_param_groups


# ============================================================
# MAIN WRAPPER
# ============================================================

@dataclass
class TrainConfig:
    cache_dir: str
    student_variant: str = "small"  # tiny|small
    mode: str = "debug"  # debug|full
    debug_train_shards: int = 2
    debug_val_shards: int = 1
    size: int = 416
    patch_size: int = 16

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
    seed: int = 42
    lambda_summary: float = 1.0
    lambda_spatial: float = 1.0
    lambda_mse: float = 0.05

    grad_checkpointing: bool = False


def pick_student_name(variant: str) -> str:
    if variant == "tiny":
        return "convnext_tiny.dinov3_lvd1689m"
    if variant == "small":
        return "convnext_small.dinov3_lvd1689m"
    raise ValueError(f"Unknown student_variant={variant}")


def infer_teacher_dims(train_ds: IterableDataset, size: int, patch_size: int) -> Tuple[int, int, int, int]:
    first_payload = None
    for _, payload in train_ds:
        first_payload = payload
        break
    if first_payload is None:
        raise RuntimeError("Could not read any samples from cache.")

    Ct = int(first_payload["summary"].shape[0])
    Tt, Dt = first_payload["spatial_tokens"].shape
    Ht = size // patch_size
    Wt = size // patch_size
    assert Tt == Ht * Wt, f"Teacher token count T={Tt} != {Ht}*{Wt} (check size/patch_size)"
    return Ct, Dt, Ht, Wt


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, Tuple[int, int, int, int], str]:
    train_shards, val_shards = find_shards(cfg.cache_dir)

    num_workers = cfg.num_workers
    eval_every = cfg.eval_every
    eval_batches = cfg.eval_batches
    persistent_workers = cfg.persistent_workers
    prefetch_factor = cfg.prefetch_factor

    if cfg.mode == "debug":
        train_shards = train_shards[: cfg.debug_train_shards]
        val_shards = val_shards[: cfg.debug_val_shards]
        num_workers = 0
        eval_every = 5
        eval_batches = 5
        persistent_workers = False
        prefetch_factor = None
        print(f"[DEBUG] Using train_shards={len(train_shards)} val_shards={len(val_shards)}")
    else:
        print(f"[FULL] Using train_shards={len(train_shards)} val_shards={len(val_shards)}")

    train_ds = TarShardDataset(train_shards, shuffle_shards=True, seed=cfg.seed)
    val_ds = TarShardDataset(val_shards, shuffle_shards=False, seed=cfg.seed)

    # teacher dims
    Ct, Dt, Ht, Wt = infer_teacher_dims(train_ds, cfg.size, cfg.patch_size)

    collate_fn = make_collate(size=cfg.size)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # return the *effective* debug overrides too
    cfg.eval_every = eval_every
    cfg.eval_batches = eval_batches
    cfg.num_workers = num_workers
    cfg.persistent_workers = persistent_workers
    cfg.prefetch_factor = prefetch_factor if prefetch_factor is not None else 2

    student_name = pick_student_name(cfg.student_variant)
    return train_dl, val_dl, (Ct, Dt, Ht, Wt), student_name


def build_model_and_optim(
        cfg: TrainConfig,
        device: str,
        Ct: int,
        Dt: int,
        Ht: int,
        Wt: int,
        student_name: str,
) -> Tuple[DistillModel, torch.optim.Optimizer, torch.amp.GradScaler]:
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

    params = []
    params += build_param_groups(model.student, cfg.wd)
    params += build_param_groups(model.sum_head, cfg.wd)
    params += build_param_groups(model.sp_head, cfg.wd)

    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    return model, opt, scaler


class DistillRunner:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        self.train_dl, self.val_dl, dims, self.student_name = build_loaders(cfg)
        self.Ct, self.Dt, self.Ht, self.Wt = dims

        self.model, self.opt, self.scaler = build_model_and_optim(
            cfg=cfg,
            device=self.device,
            Ct=self.Ct,
            Dt=self.Dt,
            Ht=self.Ht,
            Wt=self.Wt,
            student_name=self.student_name,
        )

        self.loss_w = LossWeights(
            lambda_summary=cfg.lambda_summary,
            lambda_spatial=cfg.lambda_spatial,
            lambda_mse=cfg.lambda_mse,
        )

        exp_dir = cfg.exp_root / f"exp_{time.strftime('%Y%m%d%H%M')}"
        self.tb_writer = SummaryWriter(log_dir=exp_dir / "tb")
        self.checkpoint_dir = exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        step = 0
        best_val_loss = None

        for ep in range(self.cfg.epochs):
            ep_t0 = time.time()

            step, best_val_loss = train_one_epoch(
                model=self.model,
                train_dl=self.train_dl,
                val_dl=self.val_dl,
                device=self.device,
                opt=self.opt,
                scaler=self.scaler,
                tb_writer=self.tb_writer,
                step0=step,
                ep=ep,
                amp=self.cfg.amp,
                grad_accum=self.cfg.grad_accum,
                log_every=self.cfg.log_every,
                eval_every=self.cfg.eval_every,
                eval_batches=self.cfg.eval_batches,
                Dt=self.Dt,
                Ht=self.Ht,
                Wt=self.Wt,
                w=self.loss_w,
                checkpoint_dir=self.checkpoint_dir,
                student_name=self.student_name,
                size=self.cfg.size,
                patch_size=self.cfg.patch_size,
                best_val_loss=best_val_loss,
            )

            self.tb_writer.add_scalar("epoch", ep, step)
            print(f"Epoch {ep} done in {(time.time() - ep_t0) / 60:.1f} min")


# ============================================================
# CLI
# ============================================================

def parse_args() -> TrainConfig:
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
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    ap.set_defaults(persistent_workers=True)
    ap.add_argument("--prefetch_factor", type=int, default=8)

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--wd", type=float, default=0.08)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=50)

    ap.add_argument("--exp_root", type=Path, default=Path("/home/burplord/experiments/"))
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lambda_summary", type=float, default=1.0)
    ap.add_argument("--lambda_spatial", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=0.05)

    ap.add_argument("--grad_checkpointing", action="store_true")

    a = ap.parse_args()
    return TrainConfig(**vars(a))


def main():
    cfg = parse_args()
    runner = DistillRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
