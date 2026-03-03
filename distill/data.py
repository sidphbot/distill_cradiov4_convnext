import hashlib
import io
import random
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


# ============================================================
# DATA
# ============================================================

def _stable_hash01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    return float(x)


def _stable_hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def read_image_list(list_file: str) -> List[str]:
    paths = []
    with open(list_file, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            paths.append(s)
    if not paths:
        raise RuntimeError(f"No image paths found in {list_file}")
    return paths


def split_paths_deterministic(paths: List[str], val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    train, val = [], []
    for p in paths:
        x = _stable_hash01(f"{seed}:{p}")
        (val if x < val_frac else train).append(p)
    if not train or not val:
        raise RuntimeError(f"Split produced empty train/val (val_frac={val_frac}).")
    return train, val


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.paths = list(paths)
        self.seed = 0

    def set_epoch(self, epoch: int):
        self.seed = int(epoch)
        rng = random.Random(self.seed)
        rng.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

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
        self.val_ds = None

    def setup(self, stage=None):
        all_paths = read_image_list(self.cfg.data_list)
        train_paths, val_paths = split_paths_deterministic(
            all_paths, val_frac=self.cfg.val_frac, seed=self.cfg.seed,
        )

        if getattr(self.cfg, "train_cap", 0) > 0:
            train_paths = train_paths[:self.cfg.train_cap]
        if getattr(self.cfg, "val_cap", 0) > 0:
            val_paths = val_paths[:self.cfg.val_cap]

        print(f"[{self.cfg.mode.upper()}][ONLINE] train={len(train_paths)} val={len(val_paths)}")

        self.train_ds = ImagePathDataset(train_paths)
        self.val_ds = ImagePathDataset(val_paths)

    def train_dataloader(self):
        collate = OnlineCollate(self.train_ds, size=self.cfg.size)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )

    def val_dataloader(self):
        collate = OnlineCollate(self.val_ds, size=self.cfg.size)
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )
