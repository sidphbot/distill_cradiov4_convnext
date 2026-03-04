#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
import tarfile
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, Tuple, Optional

from tqdm import tqdm
from PIL import Image as PILImage

from resize_and_save_images import _resize_save


# -----------------------
# Worker (mp)
# -----------------------
def _mp_worker_resize(item: Tuple[bytes, str, str, int, str, int]) -> Tuple[str, str, str]:
    """
    item: (img_bytes, out_dir_str, image_id, size, mode, quality)
    returns: (status, image_id, msg)
    """
    img_bytes, out_dir_str, image_id, size, mode, quality = item
    out_dir = Path(out_dir_str)
    dst = out_dir / f"{image_id}.jpg"
    try:
        with PILImage.open(io.BytesIO(img_bytes)) as im:
            im = im.convert("RGB")
            _resize_save(dst, im, mode, quality, size)
        return ("ok", image_id, "")
    except Exception as e:
        return ("fail", image_id, str(e))


# -----------------------
# Local tar iteration
# -----------------------
def _iter_imagenet_tar_members(tar_path: Path) -> Iterator[Tuple[str, bytes]]:
    """
    Yields (image_id, jpg_bytes) from an ImageNet tar.

    For train tar: contains synset sub-tars; we unpack those too.
    For val tar: usually flat JPEGs.
    """
    with tarfile.open(tar_path, "r:*") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        for m in members:
            name = Path(m.name).name
            # train tar often has nested .tar files
            if name.endswith(".tar"):
                f = tf.extractfile(m)
                if f is None:
                    continue
                nested_bytes = f.read()
                with tarfile.open(fileobj=io.BytesIO(nested_bytes), mode="r:*") as ntf:
                    for nm in ntf.getmembers():
                        if nm.isfile() and nm.name.lower().endswith((".jpg", ".jpeg")):
                            nf = ntf.extractfile(nm)
                            if nf is None:
                                continue
                            jpg = nf.read()
                            img_name = Path(nm.name).name
                            image_id = Path(img_name).stem
                            yield image_id, jpg
            else:
                if not name.lower().endswith((".jpg", ".jpeg")):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                jpg = f.read()
                image_id = Path(name).stem
                yield image_id, jpg


# -----------------------
# HF iteration
# -----------------------
def _iter_imagenet_hf(split: str, streaming: bool) -> Iterator[Tuple[str, bytes]]:
    """
    Yields (image_id, jpg_bytes) from HF datasets.

    Requires:
      - pip install datasets
      - huggingface-cli login
      - acceptance of ImageNet terms on the dataset page (gated). :contentReference[oaicite:5]{index=5}
    """
    from datasets import load_dataset  # lazy import

    # Use the canonical dataset id
    ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=streaming)

    # When streaming=True, ds is an iterable
    for idx, ex in enumerate(ds):
        img = ex["image"]  # PIL Image
        # Use stable image id: split + idx (no guaranteed original filename)
        image_id = f"{split}_{idx:09d}"
        # Encode to bytes without writing temp files
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95, optimize=True)
        yield image_id, buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--mode", choices=["crop", "pad", "squash", "longest"], default="longest")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    ap.add_argument("--chunksize", type=int, default=64)

    ap.add_argument("--source", choices=["hf", "local_tar"], required=True)

    # local tar mode
    ap.add_argument("--train_tar", type=Path, default=None, help="Path to ILSVRC2012_img_train.tar")
    ap.add_argument("--val_tar", type=Path, default=None, help="Path to ILSVRC2012_img_val.tar")
    ap.add_argument("--which", choices=["train", "val", "both"], default="both")

    # HF mode
    ap.add_argument("--streaming", action="store_true", help="Use HF streaming (recommended for low disk while downloading)")

    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def run_iter(it: Iterator[Tuple[str, bytes]], desc: str):
        ok = fail = 0
        with mp.Pool(processes=args.num_workers) as pool:
            tasks = ((jpg, str(args.output_dir), image_id, args.size, args.mode, args.quality) for (image_id, jpg) in it)
            for status, image_id, msg in tqdm(
                pool.imap_unordered(_mp_worker_resize, tasks, chunksize=args.chunksize),
                desc=desc,
            ):
                if status == "ok":
                    ok += 1
                else:
                    fail += 1
                    tqdm.write(f"[FAIL] {image_id}: {msg}")
        print(f"{desc} done. ok={ok} fail={fail}")

    if args.source == "local_tar":
        if args.which in ("train", "both"):
            if args.train_tar is None:
                raise SystemExit("--train_tar is required for --which train/both")
            run_iter(_iter_imagenet_tar_members(args.train_tar), "ImageNet train (tar) -> resized")

        if args.which in ("val", "both"):
            if args.val_tar is None:
                raise SystemExit("--val_tar is required for --which val/both")
            run_iter(_iter_imagenet_tar_members(args.val_tar), "ImageNet val (tar) -> resized")

    else:
        # HF gated: requires login + acceptance. :contentReference[oaicite:6]{index=6}
        if args.which in ("train", "both"):
            run_iter(_iter_imagenet_hf("train", streaming=args.streaming), "ImageNet train (HF) -> resized")
        if args.which in ("val", "both"):
            run_iter(_iter_imagenet_hf("validation", streaming=args.streaming), "ImageNet val (HF) -> resized")


if __name__ == "__main__":
    main()