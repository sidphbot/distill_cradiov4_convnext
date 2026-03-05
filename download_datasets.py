#!/usr/bin/env python3
"""
Download and resize COCO 2017 and/or ImageNet-1k into split directories.

Output structure:
    output_dir/
    ├── coco/
    │   ├── train/          (118k images)
    │   ├── val/            (5k images)
    │   ├── test/           (41k images)
    │   └── annotations/    (instances, captions, etc.)
    └── imagenet/
        ├── train/          (1.28M images)
        ├── val/            (50k images)
        └── annotations/    (train_labels.json, val_labels.json, class_names.json)

Usage:
    python download_datasets.py --output_dir /data --dataset all
    python download_datasets.py --output_dir /data --dataset coco --split train
    python download_datasets.py --output_dir /data --dataset imagenet --streaming
"""
from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, Tuple, Optional

import requests
from tqdm import tqdm
from PIL import Image as PILImage

from resize_and_save_images import _resize_save


# ============================================================
# Shared
# ============================================================

def _mp_worker_resize(item: Tuple[bytes, str, str, int, str, int]) -> Tuple[str, str, str]:
    img_bytes, out_dir_str, image_id, size, mode, quality = item
    dst = Path(out_dir_str) / f"{image_id}.jpg"
    try:
        with PILImage.open(io.BytesIO(img_bytes)) as im:
            im = im.convert("RGB")
            _resize_save(dst, im, mode, quality, size)
        return ("ok", image_id, "")
    except Exception as e:
        return ("fail", image_id, str(e))


def _download_file(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(dst, "wb") as f, tqdm(total=total, desc=f"Downloading {dst.name}", unit="B", unit_scale=True) as pbar:
            for b in r.iter_content(chunk_size=chunk):
                if not b:
                    continue
                f.write(b)
                if total is not None:
                    pbar.update(len(b))


def _run_resize_pool(it: Iterator[Tuple[str, bytes]], out_dir: Path, desc: str,
                     size: int, mode: str, quality: int, num_workers: int, chunksize: int) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with mp.Pool(processes=num_workers) as pool:
        tasks = ((jpg, str(out_dir), image_id, size, mode, quality) for image_id, jpg in it)
        for status, image_id, msg in tqdm(
            pool.imap_unordered(_mp_worker_resize, tasks, chunksize=chunksize),
            desc=desc,
        ):
            if status == "ok":
                ok += 1
            else:
                fail += 1
                tqdm.write(f"[FAIL] {image_id}: {msg}")
    return ok, fail


# ============================================================
# COCO
# ============================================================

COCO_ZIPS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
    "test": "http://images.cocodataset.org/zips/test2017.zip",
}

COCO_ANNOTATIONS = {
    "trainval": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "test": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
}


def _iter_coco_zip(zip_path: Path) -> Iterator[Tuple[str, bytes]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            image_id = Path(name).stem  # COCO numeric id filename
            yield image_id, zf.read(name)


def download_coco(args, coco_dir: Path, splits: list[str]) -> None:
    work_dir = args.work_dir / "coco"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Annotations
    if not args.skip_annotations:
        ann_dir = coco_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        ann_keys = set()
        if any(s in splits for s in ("train", "val")):
            ann_keys.add("trainval")
        if "test" in splits:
            ann_keys.add("test")
        for ann_key in ann_keys:
            ann_zip_path = work_dir / f"annotations_{ann_key}2017.zip"
            if not ann_zip_path.exists():
                _download_file(COCO_ANNOTATIONS[ann_key], ann_zip_path)
            with zipfile.ZipFile(ann_zip_path, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".json"):
                        dst = ann_dir / Path(name).name
                        if not dst.exists():
                            dst.write_bytes(zf.read(name))
                            print(f"Extracted {dst}")
            if not args.keep_zips:
                try:
                    ann_zip_path.unlink()
                except Exception:
                    pass

    # Images per split
    for sp in splits:
        zip_path = work_dir / f"{sp}2017.zip"
        if not zip_path.exists():
            _download_file(COCO_ZIPS[sp], zip_path)

        ok, fail = _run_resize_pool(
            _iter_coco_zip(zip_path), coco_dir / sp, f"COCO {sp} -> resized",
            args.size, args.mode, args.quality, args.num_workers, args.chunksize,
        )
        print(f"COCO {sp} done. ok={ok} fail={fail}")

        if not args.keep_zips:
            try:
                zip_path.unlink()
            except Exception:
                pass


# ============================================================
# ImageNet (HF)
# ============================================================

def _iter_imagenet_hf(split: str, streaming: bool) -> Iterator[Tuple[str, bytes, int]]:
    """Yields (image_id, jpg_bytes, label_idx) from HF datasets."""
    from datasets import load_dataset

    hf_split = "validation" if split == "val" else split
    ds = load_dataset("ILSVRC/imagenet-1k", split=hf_split, streaming=streaming)

    for idx, ex in enumerate(ds):
        img = ex["image"]
        label = ex["label"]
        image_id = f"{split}_{idx:09d}"
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95, optimize=True)
        yield image_id, buf.getvalue(), label


def _get_hf_class_names() -> Optional[list]:
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder("ILSVRC/imagenet-1k")
        return builder.info.features["label"].names
    except Exception:
        return None


def download_imagenet(args, imagenet_dir: Path, splits: list[str]) -> None:
    # Class names
    class_names = _get_hf_class_names()
    if class_names:
        cn_path = imagenet_dir / "annotations" / "class_names.json"
        cn_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cn_path, "w") as f:
            json.dump(class_names, f)
        print(f"Saved {len(class_names)} class names to {cn_path}")

    for sp in splits:
        split_dir = imagenet_dir / sp
        split_dir.mkdir(parents=True, exist_ok=True)
        labels = {}
        ok = fail = 0

        def _image_iter():
            for image_id, jpg, label in _iter_imagenet_hf(sp, streaming=args.streaming):
                labels[image_id] = label
                yield image_id, jpg

        ok, fail = _run_resize_pool(
            _image_iter(), split_dir, f"ImageNet {sp} (HF) -> resized",
            args.size, args.mode, args.quality, args.num_workers, args.chunksize,
        )
        print(f"ImageNet {sp} done. ok={ok} fail={fail}")

        if labels:
            ann_path = imagenet_dir / "annotations" / f"{sp}_labels.json"
            ann_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ann_path, "w") as f:
                json.dump(labels, f)
            print(f"Saved {len(labels)} labels to {ann_path}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Download and resize COCO and/or ImageNet")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Root output dir (datasets go into output_dir/coco/ and output_dir/imagenet/)")
    ap.add_argument("--dataset", choices=["coco", "imagenet", "all"], default="all")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                    help="Which split(s) to download. 'test' only applies to COCO.")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--mode", choices=["crop", "pad", "squash", "longest"], default="longest")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    ap.add_argument("--chunksize", type=int, default=64)
    ap.add_argument("--work_dir", type=Path, default=Path("./_download_tmp"),
                    help="Temp dir for downloaded zips")
    ap.add_argument("--keep_zips", action="store_true")
    ap.add_argument("--skip_annotations", action="store_true")
    ap.add_argument("--no-streaming", dest="streaming", action="store_false",
                    help="Disable HF streaming for ImageNet (downloads full dataset first)")
    ap.set_defaults(streaming=True)
    args = ap.parse_args()

    do_coco = args.dataset in ("coco", "all")
    do_imagenet = args.dataset in ("imagenet", "all")

    if do_coco:
        coco_splits = ["train", "val", "test"] if args.split == "all" else [args.split]
        coco_dir = args.output_dir / "coco"
        coco_dir.mkdir(parents=True, exist_ok=True)
        download_coco(args, coco_dir, coco_splits)

    if do_imagenet:
        # ImageNet has no test split
        if args.split == "all":
            in_splits = ["train", "val"]
        elif args.split == "test":
            print("ImageNet has no public test split, skipping.")
            in_splits = []
        else:
            in_splits = [args.split]
        if in_splits:
            imagenet_dir = args.output_dir / "imagenet"
            imagenet_dir.mkdir(parents=True, exist_ok=True)
            download_imagenet(args, imagenet_dir, in_splits)


if __name__ == "__main__":
    main()
