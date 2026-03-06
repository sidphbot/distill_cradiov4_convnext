"""Data preprocessing utilities — re-exports from distill.data + helpers."""

import torch
from torch.utils.data import DataLoader

from distill.data import (
    _pad_resize as pad_resize,
    _collect_images as collect_images,
    OnlineCollate,
    ImagePathDataset,
)
from bench.common.config import IMAGENET_MEAN, IMAGENET_STD

# Re-export for clean public API
__all__ = [
    "pad_resize",
    "collect_images",
    "OnlineCollate",
    "ImagePathDataset",
    "normalize_batch",
    "make_eval_dataloader",
]


def normalize_batch(x: torch.Tensor,
                    mean: tuple = IMAGENET_MEAN,
                    std: tuple = IMAGENET_STD) -> torch.Tensor:
    """Normalize (B,3,H,W) float [0,1] tensor with ImageNet stats."""
    m = torch.tensor(list(mean), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(list(std), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return x.sub_(m).div_(s)


def make_eval_dataloader(paths: list,
                         size: int = 416,
                         batch_size: int = 32,
                         num_workers: int = 8) -> DataLoader:
    """Create a deterministic evaluation DataLoader."""
    ds = ImagePathDataset(paths)
    ds.seed = 0  # fixed for reproducibility
    collate = OnlineCollate(ds, size=size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )
