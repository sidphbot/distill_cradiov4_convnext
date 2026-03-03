import glob
import io
import os
import random
import tarfile
from PIL import Image
import hashlib
from typing import List, Tuple

import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import IterableDataset
from torchvision.io import decode_jpeg
from torchvision.transforms import InterpolationMode


# ============================================================
# DATA
# ============================================================

def _stable_hash01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    return float(x)

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
    # Deterministic per-path split; seed just offsets the hash slightly
    train, val = [], []
    for p in paths:
        x = _stable_hash01(f"{seed}:{p}")
        (val if x < val_frac else train).append(p)
    if not train or not val:
        raise RuntimeError(f"Split produced empty train/val (val_frac={val_frac}).")
    return train, val

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, paths: List[str]):
        self.paths = list(paths)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        rng = random.Random(self.epoch)
        rng.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        return p


def make_online_collate(size: int = 416, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Returns:
      x: (B,3,size,size) float32 normalized (student input)
      pil_rs: List[PIL.Image] resized to (size,size) for teacher proc(do_resize=False)
      keys: List[str] (we use the path as key; if you want basename-only, change here)
    """
    def _collate(batch):
        xs = []
        pil_rs = []
        keys = []

        for p in batch:
            try:
                with open(p, "rb") as f:
                    b = f.read()
                pil = Image.open(io.BytesIO(b)).convert("RGB")
                pil = pil.resize((size, size), resample=Image.BICUBIC)  # EXACTLY like caching
                # student tensor from resized PIL (no extra resize)
                img_u8 = TF.pil_to_tensor(pil)  # (C,H,W) uint8
                x = img_u8.float().div(255.0)
                x = TF.normalize(x, mean=mean, std=std)
                xs.append(x)
                pil_rs.append(pil)
                keys.append(p)
            except Exception:
                continue

        if not xs:
            # In case a whole batch fails decoding
            raise RuntimeError("All samples in batch failed to load.")

        return torch.stack(xs, dim=0), pil_rs, keys

    return _collate

def decode_image_bytes_fast(img_bytes: bytes) -> torch.Tensor:
    # returns uint8 tensor (C,H,W)
    buf = torch.frombuffer(img_bytes, dtype=torch.uint8)
    img = decode_jpeg(buf, device="cpu", mode=torchvision.io.ImageReadMode.RGB)  # (C,H,W) uint8
    return img


def preprocess_uint8(
        img_u8: torch.Tensor,
        size: int = 416,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
) -> torch.Tensor:
    # img_u8: (C,H,W) uint8
    x = img_u8.float().div(255.0)
    x = TF.resize(x, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    x = TF.normalize(x, mean=mean, std=std)
    return x


class TarShardDataset(IterableDataset):
    """
    Streams samples from a list of .tar shards.
    Each sample has:
      {key}.img  (raw image bytes)
      {key}.pt   (torch payload dict with summary + spatial_tokens)
    """

    def __init__(self, shard_paths: List[str], shuffle_shards: bool = True, seed: int = 0):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _iter_tar(self, tar_path: str):
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            imgs = {m.name[:-4] for m in members if m.name.endswith(".img")}
            pts = {m.name[:-3] for m in members if m.name.endswith(".pt")}
            keys = sorted(list(imgs.intersection(pts)))

            for k in keys:
                try:
                    img_m = tf.getmember(k + ".img")
                    pt_m = tf.getmember(k + ".pt")

                    img_f = tf.extractfile(img_m)
                    pt_f = tf.extractfile(pt_m)
                    if img_f is None or pt_f is None:
                        continue

                    img_bytes = img_f.read()

                    # IMPORTANT: buffer the .pt into BytesIO before torch.load
                    pt_bytes = pt_f.read()
                    payload = torch.load(io.BytesIO(pt_bytes), map_location="cpu", weights_only=False)

                    yield img_bytes, payload, k
                except Exception:
                    continue

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        shard_paths = self.shard_paths

        if self.shuffle_shards:
            wid = worker.id if worker else 0
            rng = random.Random(self.seed + 1000 * self.epoch + wid)
            rng.shuffle(shard_paths)

        if worker is not None:
            shard_paths = shard_paths[worker.id:: worker.num_workers]

        for sp in shard_paths:
            yield from self._iter_tar(sp)


def make_collate(size: int = 416):
    def _collate(batch):
        xs, t_summ, t_spat, keys = [], [], [], []
        for img_bytes, payload, key in batch:
            img_u8 = decode_image_bytes_fast(img_bytes)
            xs.append(preprocess_uint8(img_u8, size=size))
            t_summ.append(payload["summary"])
            t_spat.append(payload["spatial_tokens"])
            keys.append(key)
        return torch.stack(xs), torch.stack(t_summ), torch.stack(t_spat), keys

    return _collate


def find_shards(cache_dir: str) -> Tuple[List[str], List[str]]:
    train_shards = sorted(glob.glob(os.path.join(cache_dir, "train", "*.tar")))
    val_shards = sorted(glob.glob(os.path.join(cache_dir, "val", "*.tar")))
    if not train_shards or not val_shards:
        raise RuntimeError("No shards found. Expected train/*.tar and val/*.tar under cache_dir")
    return train_shards, val_shards
