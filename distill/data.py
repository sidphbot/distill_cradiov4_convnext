import glob
import io
import os
import random
import tarfile
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

                    yield img_bytes, payload
                except Exception:
                    continue

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        shard_paths = self.shard_paths

        if self.shuffle_shards:
            rng = random.Random(self.seed + (worker.id if worker else 0))
            rng.shuffle(shard_paths)

        if worker is not None:
            shard_paths = shard_paths[worker.id:: worker.num_workers]

        for sp in shard_paths:
            yield from self._iter_tar(sp)


def make_collate(size: int = 416):
    def _collate(batch):
        xs, t_summ, t_spat = [], [], []
        for img_bytes, payload in batch:
            img_u8 = decode_image_bytes_fast(img_bytes)
            xs.append(preprocess_uint8(img_u8, size=size))
            t_summ.append(payload["summary"])
            t_spat.append(payload["spatial_tokens"])
        return torch.stack(xs), torch.stack(t_summ), torch.stack(t_spat)

    return _collate


def find_shards(cache_dir: str) -> Tuple[List[str], List[str]]:
    train_shards = sorted(glob.glob(os.path.join(cache_dir, "train", "*.tar")))
    val_shards = sorted(glob.glob(os.path.join(cache_dir, "val", "*.tar")))
    if not train_shards or not val_shards:
        raise RuntimeError("No shards found. Expected train/*.tar and val/*.tar under cache_dir")
    return train_shards, val_shards
