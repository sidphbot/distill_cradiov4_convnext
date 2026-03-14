import hashlib
import io
import os
import random
from typing import Dict, List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


# ============================================================
# DATA
# ============================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _stable_hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _stable_hash01(s: str) -> float:
    """Return a deterministic float in [0, 1) for a string key."""
    return _stable_hash_int(s) / 0x1_0000_0000


def _collect_images(directory: str) -> List[str]:
    """Recursively collect image paths from a directory."""
    root = os.path.abspath(directory)
    paths = sorted(
        os.path.join(dp, f)
        for dp, _, fnames in os.walk(root)
        for f in fnames
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    return paths


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], n: int = None):
        self.paths = list(paths)
        self.seed = 0
        self.n = n

    def set_epoch(self, epoch: int):
        self.seed = int(epoch)*random.randint(0, 5000)
        rng = random.Random(self.seed)
        rng.shuffle(self.paths)

    def __len__(self):
        return len(self.paths) if not self.n else self.n

    def __getitem__(self, idx: int):
        return self.paths[idx]


def _pad_resize(pil: Image.Image, size: int) -> Image.Image:
    """Resize preserving aspect ratio, pad shorter side with black to (size, size)."""
    w, h = pil.size
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pil = pil.resize((new_w, new_h), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    canvas.paste(pil, (paste_x, paste_y))
    return canvas


class OnlineCollate:
    """
    Callable collate that reads the dataset's seed to deterministically
    choose pad-or-squash per image (50/50 based on stable hash).

    Returns:
        img_u8: (B, 3, H, W) uint8 tensor (NOT normalized)
        pil_rs: List[PIL.Image] resized to (size, size) — for teacher
        keys:   List[str] image paths
    """

    def __init__(self, dataset: ImagePathDataset, size: int = 416):
        self.dataset = dataset
        self.size = size

    def __call__(self, batch):
        xs = []
        pil_rs = []
        keys = []
        seed = self.dataset.seed

        for p in batch:
            try:
                with open(p, "rb") as f:
                    b = f.read()
                pil = Image.open(io.BytesIO(b)).convert("RGB")

                # Deterministic pad-or-squash decision per (seed, path)
                h = _stable_hash_int(f"{seed}:{p}")
                use_pad = (h % 2) == 0

                if use_pad:
                    pil = _pad_resize(pil, self.size)
                else:
                    pil = pil.resize((self.size, self.size), resample=Image.BICUBIC)

                img_u8 = TF.pil_to_tensor(pil)  # (C, H, W) uint8
                xs.append(img_u8)
                pil_rs.append(pil)
                keys.append(p)
            except Exception:
                continue

        if not xs:
            raise RuntimeError("All samples in batch failed to load.")

        return torch.stack(xs, dim=0), pil_rs, keys


class DistillDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_datasets: Dict[str, ImagePathDataset] = {}
        self.val_source_names: List[str] = []

    def setup(self, stage=None):
        val_frac = getattr(self.cfg.data, "val_frac", 0.02)
        val_cap = getattr(self.cfg.data, "val_cap", 0)

        # --- OI: split image_dir into train + oi_val ---
        all_oi = _collect_images(self.cfg.data.image_dir)
        if not all_oi:
            raise RuntimeError(f"No images found in image_dir: {self.cfg.data.image_dir}")

        oi_train = []
        oi_val = []
        for p in all_oi:
            if _stable_hash01(p) < val_frac:
                oi_val.append(p)
            else:
                oi_train.append(p)

        if not oi_val:
            raise RuntimeError(
                f"val_frac={val_frac} produced 0 val images from {len(all_oi)} total. Increase val_frac."
            )

        # Apply caps
        train_cap = getattr(self.cfg.data, "train_cap", 0)
        self.train_ds = ImagePathDataset(oi_train, n=train_cap)

        # Build ordered val datasets: oi_val first, then config val_sources
        self.val_datasets = {}
        self.val_source_names = []

        self.val_datasets["oi_val"] = ImagePathDataset(oi_val, n=val_cap)
        self.val_source_names.append("oi_val")

        # --- Off-distribution val sources ---
        val_sources = getattr(self.cfg.data, "val_sources", {}) or {}
        for name, directory in val_sources.items():
            if not directory:
                continue
            vp = _collect_images(directory)
            if not vp:
                print(f"[WARN] No images found in val source '{name}': {directory}")
                continue
            self.val_datasets[name] = ImagePathDataset(vp, n=val_cap)
            self.val_source_names.append(name)

        # Summary
        parts = [f"{name}={len(ds)}" for name, ds in self.val_datasets.items()]
        print(
            f"[DATA] train={len(self.train_ds)} (OI, val_frac={val_frac}) "
            f"val: {', '.join(parts)}"
        )

    def train_dataloader(self):
        collate = OnlineCollate(self.train_ds, size=self.cfg.data.size)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=True,
            collate_fn=collate,
            persistent_workers=self.cfg.dataloader.persistent_workers and self.cfg.dataloader.num_workers > 0,
            prefetch_factor=self.cfg.dataloader.prefetch_factor if self.cfg.dataloader.num_workers > 0 else None,
        )

    def val_dataloader(self):
        """Return a list of DataLoaders — one per val source, in val_source_names order."""
        loaders = []
        for name in self.val_source_names:
            ds = self.val_datasets[name]
            collate = OnlineCollate(ds, size=self.cfg.data.size)
            loaders.append(DataLoader(
                ds,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=False,
                num_workers=self.cfg.dataloader.num_workers,
                pin_memory=True,
                collate_fn=collate,
                persistent_workers=self.cfg.dataloader.persistent_workers and self.cfg.dataloader.num_workers > 0,
                prefetch_factor=self.cfg.dataloader.prefetch_factor if self.cfg.dataloader.num_workers > 0 else None,
            ))
        return loaders
