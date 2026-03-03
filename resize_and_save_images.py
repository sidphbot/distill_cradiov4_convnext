#!/usr/bin/env python3
"""
Resize images from input_dir into output_dir as {image_id}.jpg, squared to 416x416.

Modes:
- crop   : center-crop to square, then resize
- pad    : pad to square, then resize
- squash : direct resize to square (aspect ratio distortion)

Install:
  pip install pillow tqdm
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace
from pathlib import Path
import multiprocessing as mp
from typing import Tuple

from PIL import Image
from tqdm import tqdm


def resize_longest(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    if w > h:
        left = (w - h) // 2
        return img.crop((left, 0, left + h, h))
    else:
        top = (h - w) // 2
        return img.crop((0, top, w, top + w))


def square_pad(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    out = Image.new("RGB", (s, s), (0, 0, 0))
    x = (s - w) // 2
    y = (s - h) // 2
    out.paste(img, (x, y))
    return out


def _resize_save(dst_path: Path, im: Image.Image, mode: str, quality: int, size: int):
    im = im.convert("RGB")

    if mode == "crop":
        im = square_crop(im)
        im = im.resize((size, size), resample=Image.BICUBIC)

    elif mode == "pad":
        im = square_pad(im)
        im = im.resize((size, size), resample=Image.BICUBIC)

    elif mode == "squash":
        im = im.resize((size, size), resample=Image.BICUBIC)

    elif mode == "longest":
        im = resize_longest(im, size)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    im.save(dst_path, format="JPEG", quality=quality, optimize=True)


def process_one(src_path: Path, dst_path: Path, size: int, mode: str, quality: int) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as im:
        _resize_save(dst_path, im, mode, quality, size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument(
        "--mode",
        choices=["crop", "pad", "squash", "longest"],
        default="longest",
    )
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = ap.parse_args()

    in_dir: Path = args.input_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    srcs = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if not srcs:
        raise SystemExit(f"No images found in {in_dir}")

    # pass only primitives + paths as strings
    worker_args = (str(args.output_dir), args.size, args.mode, args.quality, args.overwrite)

    with mp.Pool(processes=args.num_workers) as pool:
        # each item is a single tuple -> worker unpacks
        it = ((worker_args, str(src)) for src in srcs)
        for status, name, msg in tqdm(pool.imap_unordered(download_worker, it, chunksize=64), total=len(srcs)):
            if status == "warn":
                print(f"[WARN] {name}: {msg}")

    print("Done.")


def download_worker(item: Tuple[Tuple[str, int, str, int, bool], str]):
    (out_dir_str, size, mode, quality, overwrite), src_str = item
    src = Path(src_str)

    image_id = src.stem
    dst = Path(out_dir_str) / f"{image_id}.jpg"

    if dst.exists() and not overwrite:
        return ("skip", src.name, "")

    try:
        # If you care about race safety, write temp then rename (see below)
        process_one(src, dst, size, mode, quality)
        return ("ok", src.name, "")
    except Exception as e:
        return ("warn", src.name, str(e))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=False)  # optional; see note below
    main()