#!/usr/bin/env python3
# cache_cradiov4h_openimages_candidates_async.py
#
# Deterministic incremental caching via a growing candidate list of OpenImages IDs.
#
# Key features:
# - Reads a user-managed candidate list file of Open Images IDs (image_ids.txt)
# - Loads ONLY those IDs via FiftyOne Open Images V6 zoo (image_ids=...)
# - Append-only shards (never rewrites existing shards)
# - Auto-resume shard indices per split by scanning existing shard filenames
# - Optional skip-cached: reads cached_ids file and only caches newly appended IDs
# - Writes shards to NVMe; optionally async copies to HDD and can delete NVMe shard after copy
# - Throttle max pending copies, plus NVMe free-space guard
#
# Shard layout:
#   nvme_out_dir/{train,val,test}/train-00000.tar ...
# Each sample in tar:
#   {image_id}.img  (original image bytes)
#   {image_id}.pt   (torch payload dict)

import os
import re
import io
import json
import time
import tarfile
import hashlib
import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, Future

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import AutoModel, CLIPImageProcessor
import fiftyone.zoo as foz


# ---------------- deterministic split ----------------
def stable_split(key: str, train_pct: float, val_pct: float) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < train_pct:
        return "train"
    elif x < train_pct + val_pct:
        return "val"
    return "test"


# ---------------- helpers ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def file_to_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def sanitize_key(s: str) -> str:
    # OpenImages IDs are safe, but sanitize path separators just in case
    return s.replace("/", "_").replace("\\", "_").strip()


def nvme_free_gb(path: str) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def guard_nvme_free_space(nvme_root: str, min_free_gb: float, copy_mgr, sleep_s: float = 1.0):
    if min_free_gb <= 0:
        return
    while nvme_free_gb(nvme_root) < min_free_gb:
        copy_mgr.poll()
        free_now = nvme_free_gb(nvme_root)
        print(f"[NVME-GUARD] free={free_now:.1f}GB < {min_free_gb:.1f}GB; waiting...")
        time.sleep(sleep_s)


def throttle_if_needed(copy_mgr, max_pending: int, sleep_s: float = 0.25):
    if (not copy_mgr.enabled) or (max_pending <= 0):
        return
    while len(copy_mgr.futures) >= max_pending:
        copy_mgr.poll()
        time.sleep(sleep_s)


# ---------------- candidate list ----------------
def read_candidate_ids(candidate_file: str) -> List[str]:
    """
    Supports lines like:
      <image-id>
      train/<image-id>   (we strip split prefixes if present)
    """
    ids = []
    with open(candidate_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # strip optional "split/" prefix
            if "/" in s:
                parts = s.split("/", 1)
                # if it's one of known splits, strip it
                if parts[0] in ("train", "validation", "test"):
                    s = parts[1]
            ids.append(s)
    return ids


def load_cached_id_set(cached_ids_path: str) -> set:
    """
    Loads cached image IDs from a newline-separated file.
    For 500k IDs this is totally fine on 64GB RAM.
    """
    if not os.path.exists(cached_ids_path):
        return set()
    out = set()
    with open(cached_ids_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.add(s)
    return out


def append_cached_id(cached_ids_path: str, image_id: str):
    # append-only log
    with open(cached_ids_path, "a", encoding="utf-8") as f:
        f.write(image_id + "\n")


# ---------------- shard index auto-resume ----------------
_SHARD_RE = re.compile(r"^(train|val|test)-(\d{5})\.tar$")

def find_next_shard_index(split_dir: str, split_name: str) -> int:
    """
    Scans split_dir for {split}-{idx:05d}.tar and returns max+1.
    """
    ensure_dir(split_dir)
    max_idx = -1
    prefix = split_name + "-"
    for fn in os.listdir(split_dir):
        if not fn.startswith(prefix) or not fn.endswith(".tar"):
            continue
        m = _SHARD_RE.match(fn)
        if not m:
            continue
        idx = int(m.group(2))
        max_idx = max(max_idx, idx)
    return max_idx + 1


# ---------------- torch dataset ----------------
class KeyPathDataset(Dataset):
    def __init__(self, keys: List[str], paths: List[str]):
        assert len(keys) == len(paths)
        self.keys = keys
        self.paths = paths

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.keys[idx], self.paths[idx]


def collate_load(batch: List[Tuple[str, str]]):
    """
    Runs inside DataLoader workers:
      - reads original image bytes
      - decodes PIL from bytes
    """
    keys, paths = zip(*batch)
    out_keys, out_paths = [], []
    img_bytes_list, pil_list = [], []

    for k, pth in zip(keys, paths):
        try:
            b = file_to_bytes(pth)
            pil = Image.open(io.BytesIO(b)).convert("RGB")
            out_keys.append(k)
            out_paths.append(pth)
            img_bytes_list.append(b)
            pil_list.append(pil)
        except Exception:
            continue

    return out_keys, out_paths, img_bytes_list, pil_list


# ---------------- teacher forward ----------------
@torch.no_grad()
def teacher_forward_fixed(teacher, proc, pil_imgs: List[Image.Image], device: str, size: int = 416):
    # Force exactly size x size
    pil_rs = [im.resize((size, size), resample=Image.BICUBIC) for im in pil_imgs]
    pv = proc(images=pil_rs, return_tensors="pt", do_resize=False, do_center_crop=False).pixel_values.to(device)
    summary, spatial = teacher(pv)  # summary (B,Ct)  spatial_tokens (B,T,Dt)
    return summary, spatial


# ---------------- async copy manager ----------------
class AsyncCopyManager:
    def __init__(self, enabled: bool, hdd_root: Optional[str], workers: int = 2, delete_src: bool = False):
        self.enabled = enabled
        self.hdd_root = hdd_root
        self.delete_src = delete_src
        self.pool = ThreadPoolExecutor(max_workers=workers) if enabled else None
        self.futures: List[Future] = []
        self.copied_ok = 0
        self.copied_fail = 0

    def submit_copy(self, src_path: str, rel_path_from_nvme_root: str):
        if not self.enabled:
            return

        dst_path = os.path.join(self.hdd_root, rel_path_from_nvme_root)
        ensure_dir(os.path.dirname(dst_path))

        def _copy():
            tmp_dst = dst_path + ".tmp"
            try:
                shutil.copy2(src_path, tmp_dst)
                os.replace(tmp_dst, dst_path)  # atomic rename
                if self.delete_src:
                    os.remove(src_path)
                return True, src_path, dst_path, None
            except Exception as e:
                try:
                    if os.path.exists(tmp_dst):
                        os.remove(tmp_dst)
                except Exception:
                    pass
                return False, src_path, dst_path, str(e)

        self.futures.append(self.pool.submit(_copy))

    def poll(self):
        if not self.enabled:
            return
        still = []
        for f in self.futures:
            if f.done():
                ok, src, dst, err = f.result()
                if ok:
                    self.copied_ok += 1
                else:
                    self.copied_fail += 1
                    print(f"[COPY-ERR] {src} -> {dst} : {err}")
            else:
                still.append(f)
        self.futures = still

    def finalize(self):
        if not self.enabled:
            return
        for f in self.futures:
            ok, src, dst, err = f.result()
            if ok:
                self.copied_ok += 1
            else:
                self.copied_fail += 1
                print(f"[COPY-ERR] {src} -> {dst} : {err}")
        self.futures = []
        self.pool.shutdown(wait=True)
        print(f"[COPY] done ok={self.copied_ok} fail={self.copied_fail}")


# ---------------- shard writer (append-only + atomic finalize) ----------------
@dataclass
class ShardWriter:
    nvme_root: str
    split: str
    shard_size: int
    copy_mgr: AsyncCopyManager
    shard_idx: int

    count_in_shard: int = 0
    tar: Optional[tarfile.TarFile] = None
    current_tmp_path: Optional[str] = None
    current_final_path: Optional[str] = None

    def _final_tar_path(self) -> str:
        ensure_dir(os.path.join(self.nvme_root, self.split))
        return os.path.join(self.nvme_root, self.split, f"{self.split}-{self.shard_idx:05d}.tar")

    def _tmp_tar_path(self, final_path: str) -> str:
        return final_path + ".tmp"

    def _rel_path(self, abs_nvme_path: str) -> str:
        return os.path.relpath(abs_nvme_path, start=self.nvme_root)

    def _open_new(self):
        # close previous shard (finalize + enqueue copy)
        if self.tar is not None:
            self._close_finalize_enqueue()

        self.count_in_shard = 0
        self.current_final_path = self._final_tar_path()
        self.current_tmp_path = self._tmp_tar_path(self.current_final_path)

        # Write tar to tmp first (crash-safe)
        self.tar = tarfile.open(self.current_tmp_path, "w")
        meta = {"split": self.split, "shard_idx": self.shard_idx, "created_at": time.time()}
        self._add_bytes(f"__meta__-{self.shard_idx:05d}.json", json.dumps(meta).encode("utf-8"))

    def _add_bytes(self, name: str, data: bytes):
        ti = tarfile.TarInfo(name=name)
        ti.size = len(data)
        ti.mtime = int(time.time())
        self.tar.addfile(ti, io.BytesIO(data))

    def add_sample(self, key: str, img_bytes: bytes, payload: dict):
        if self.tar is None or self.count_in_shard >= self.shard_size:
            if self.tar is None:
                self._open_new()
            else:
                self.shard_idx += 1
                self._open_new()

        self._add_bytes(f"{key}.img", img_bytes)
        buf = io.BytesIO()
        torch.save(payload, buf)
        self._add_bytes(f"{key}.pt", buf.getvalue())
        self.count_in_shard += 1

    def _close_finalize_enqueue(self):
        assert self.tar is not None and self.current_tmp_path and self.current_final_path
        self.tar.close()
        # atomically finalize the shard name
        os.replace(self.current_tmp_path, self.current_final_path)

        final_path = self.current_final_path
        # reset state
        self.tar = None
        self.current_tmp_path = None
        self.current_final_path = None

        # enqueue background copy of the finalized .tar
        rel = self._rel_path(final_path)
        self.copy_mgr.submit_copy(final_path, rel)

    def close(self):
        if self.tar is not None:
            self._close_finalize_enqueue()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--teacher_id", type=str, default="nvidia/C-RADIOv4-H")

    ap.add_argument("--candidate_file", type=str, required=True,
                    help="Growing list of OpenImages IDs (one per line). You append more IDs and rerun.")
    ap.add_argument("--nvme_out_dir", type=str, required=True)
    ap.add_argument("--hdd_out_dir", type=str, default=None)

    ap.add_argument("--cached_ids_file", type=str, default=None,
                    help="If not provided, defaults to <nvme_out_dir>/cached_image_ids.txt")
    ap.add_argument("--skip_cached", action="store_true",
                    help="Skip image_ids already present in cached_ids_file (recommended for incremental runs)")

    ap.add_argument("--train_pct", type=float, default=0.90)
    ap.add_argument("--val_pct", type=float, default=0.05)

    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", type=bool, default=True)

    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--fp16", type=bool, default=True)
    ap.add_argument("--amp", type=bool, default=True)

    ap.add_argument("--copy_workers", type=int, default=2)
    ap.add_argument("--delete_nvme_after_copy", type=bool, default=True)
    ap.add_argument("--max_pending_copies", type=int, default=5,
                    help="Throttle caching if background copies backlog exceeds this. Set 0 to disable.")
    ap.add_argument("--min_free_nvme_gb", type=float, default=150.0,
                    help="Pause caching if NVMe free space falls below this (GB). Set 0 to disable.")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert args.train_pct + args.val_pct < 1.0, "train_pct + val_pct must be < 1"

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    print(f"Device: {device}")

    ensure_dir(args.nvme_out_dir)
    if args.hdd_out_dir:
        ensure_dir(args.hdd_out_dir)

    cached_ids_file = args.cached_ids_file or os.path.join(args.nvme_out_dir, "cached_image_ids.txt")

    # Load candidate ids
    candidate_ids = read_candidate_ids(args.candidate_file)
    print(f"Candidate IDs: {len(candidate_ids)} from {args.candidate_file}")

    cached_set = set()
    if args.skip_cached:
        cached_set = load_cached_id_set(cached_ids_file)
        print(f"Loaded cached IDs: {len(cached_set)} from {cached_ids_file}")

    # Filter candidates (incremental)
    if args.skip_cached and cached_set:
        before = len(candidate_ids)
        candidate_ids = [cid for cid in candidate_ids if cid not in cached_set]
        print(f"After skip_cached: {len(candidate_ids)} (skipped {before - len(candidate_ids)})")

    if not candidate_ids:
        print("Nothing to do (no new candidate IDs). Exiting.")
        return

    # Teacher
    proc = CLIPImageProcessor.from_pretrained(args.teacher_id)
    teacher = AutoModel.from_pretrained(args.teacher_id, trust_remote_code=True).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # # Load ONLY these IDs from OpenImages via FiftyOne
    # # FiftyOne supports providing a TXT/JSON/CSV file to image_ids parameter.
    # # We'll pass your candidate file directly for determinism.
    # print("Loading Open Images V6 subset via FiftyOne using image_ids=... (deterministic)...")
    # fo_ds = foz.load_zoo_dataset(
    #     "open-images-v6",
    #     splits=None,
    #     split=None,
    #     label_types=[],
    #     image_ids=args.candidate_file,   # <--- deterministic list file
    #     include_id=True,
    #     shuffle=False,
    # )

    # # Build a mapping from OpenImages ID -> filepath
    # # We trust FO to give us filepaths for requested image_ids.
    # # Key = filename stem is commonly the OpenImages ID; we also try include_id field.
    # # But since we control the candidate list, we’ll match by filename stem first,
    # # and fall back to sample.filepath stem if needed.
    # id_to_path: Dict[str, str] = {}
    # for s in fo_ds.iter_samples(progress=True):
    #     # Most reliable: filename stem often equals OpenImages ID
    #     stem = sanitize_key(os.path.splitext(os.path.basename(s.filepath))[0])
    #     id_to_path[stem] = s.filepath
    #     # Also store FO internal id (not used as key)
    #     # (We don't need it for deterministic caching.)

    # Now create list in the same order as candidate_ids (deterministic)
    keys = []
    paths = []
    missing = 0
    data_path = Path('/home/burplord/fiftyone/open-images-v6/train/data')
    for cid in candidate_ids:
        p = data_path / f'{cid}.jpg'
        if p is None or not p.exists():
            # Sometimes candidate file might contain split/imageid format that was stripped,
            # or FO might store with different naming. Try a looser match:
            # (not ideal, but better than failing hard)
            missing += 1
            continue
        keys.append(cid)
        paths.append(p)

    if missing:
        print(f"[WARN] {missing} candidate IDs not found in loaded dataset; they will be skipped.")

    print(f"Resolved {len(keys)} candidates to local filepaths.")

    # Auto-resume shard indices per split
    next_train = find_next_shard_index(os.path.join(args.hdd_out_dir, "train"), "train")
    next_val   = find_next_shard_index(os.path.join(args.hdd_out_dir, "val"), "val")
    next_test  = find_next_shard_index(os.path.join(args.hdd_out_dir, "test"), "test")
    print(f"Auto-resume shard idx: train={next_train} val={next_val} test={next_test}")

    copy_mgr = AsyncCopyManager(
        enabled=bool(args.hdd_out_dir),
        hdd_root=args.hdd_out_dir,
        workers=args.copy_workers,
        delete_src=args.delete_nvme_after_copy,
    )

    writers: Dict[str, ShardWriter] = {
        "train": ShardWriter(args.nvme_out_dir, "train", args.shard_size, copy_mgr, shard_idx=next_train),
        "val":   ShardWriter(args.nvme_out_dir, "val",   args.shard_size, copy_mgr, shard_idx=next_val),
        "test":  ShardWriter(args.nvme_out_dir, "test",  args.shard_size, copy_mgr, shard_idx=next_test),
    }

    ds = KeyPathDataset(keys, paths)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,  # deterministic ordering
        num_workers=args.num_workers,
        collate_fn=collate_load,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        drop_last=False,
    )

    total = 0
    t0 = time.time()

    for batch_idx, (b_keys, b_paths, b_img_bytes, b_pils) in tqdm(enumerate(dl)):
        if not b_keys:
            continue

        guard_nvme_free_space(args.nvme_out_dir, args.min_free_nvme_gb, copy_mgr, sleep_s=1.0)
        throttle_if_needed(copy_mgr, args.max_pending_copies, sleep_s=0.25)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(args.amp and device.startswith("cuda"))):
                summary, spatial = teacher_forward_fixed(teacher, proc, b_pils, device=device, size=args.size)

        if args.fp16:
            summary = summary.to(torch.float16).cpu()
            spatial = spatial.to(torch.float16).cpu()
        else:
            summary = summary.cpu()
            spatial = spatial.cpu()

        for j, key in enumerate(b_keys):
            # skip cached safety (in case candidate file accidentally repeats IDs)
            if args.skip_cached and key in cached_set:
                continue

            split = stable_split(key, args.train_pct, args.val_pct)
            payload = {
                "image_id": key,
                "path": b_paths[j],
                "size": args.size,
                "summary": summary[j].detach().contiguous().clone(),
                "spatial_tokens": spatial[j].detach().contiguous().clone(),
            }
            tensor_mb = (
                                payload["summary"].numel() * payload["summary"].element_size()
                                + payload["spatial_tokens"].numel() * payload["spatial_tokens"].element_size()
                        ) / (1024 ** 2)

            if tensor_mb > 10:
                raise RuntimeError(f"Unexpected tensor size {tensor_mb:.2f}MB for {key} (should be ~1.7MB)")
            writers[split].add_sample(key, b_img_bytes[j], payload)

            # record cached id
            append_cached_id(cached_ids_file, key)
            if args.skip_cached:
                cached_set.add(key)

            total += 1

        if (batch_idx + 1) % 10 == 0:
            copy_mgr.poll()

        if total % 5000 == 0:
            dt = time.time() - t0
            free_gb = nvme_free_gb(args.nvme_out_dir)
            print(
                f"cached_new={total} imgs/sec={total/max(dt,1e-6):.2f} "
                f"nvme_free={free_gb:.1f}GB "
                f"pending_copies={len(copy_mgr.futures)} ok={copy_mgr.copied_ok} fail={copy_mgr.copied_fail}"
            )

    for w in writers.values():
        w.close()

    copy_mgr.finalize()

    dt = time.time() - t0
    print(f"Done. cached_new={total} time={dt/60:.1f}min avg imgs/sec={total/max(dt,1e-6):.2f}")
    print(f"cached_ids_file: {cached_ids_file}")
    print(f"NVMe shards: {args.nvme_out_dir}")
    if args.hdd_out_dir:
        print(f"HDD mirror: {args.hdd_out_dir}")
        if args.delete_nvme_after_copy:
            print("NVMe shards deleted after successful copy.")


if __name__ == "__main__":
    main()
