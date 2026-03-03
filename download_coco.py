#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
import zipfile
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, Tuple

import requests
from tqdm import tqdm
from PIL import Image as PILImage

from resize_and_save_images import _resize_save


COCO_ZIPS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
    "test": "http://images.cocodataset.org/zips/test2017.zip",
}


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


def _iter_zip_images(zip_path: Path) -> Iterator[Tuple[str, bytes]]:
    # COCO zip contains folder train2017/*.jpg etc
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            image_id = Path(name).stem  # COCO uses numeric id-like filename
            yield image_id, zf.read(name)


def _mp_worker_resize(item: Tuple[bytes, str, str, int, str, int]) -> Tuple[str, str, str]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--work_dir", type=Path, default=Path("./_coco_tmp"))
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--mode", choices=["crop", "pad", "squash", "longest"], default="longest")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    ap.add_argument("--chunksize", type=int, default=64)
    ap.add_argument("--keep_zips", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for sp in splits:
        url = COCO_ZIPS[sp]
        zip_path = args.work_dir / f"{sp}2017.zip"

        if not zip_path.exists():
            _download_file(url, zip_path)

        it = _iter_zip_images(zip_path)

        ok = fail = 0
        with mp.Pool(processes=args.num_workers) as pool:
            tasks = ((jpg, str(args.output_dir), image_id, args.size, args.mode, args.quality) for (image_id, jpg) in it)
            for status, image_id, msg in tqdm(
                pool.imap_unordered(_mp_worker_resize, tasks, chunksize=args.chunksize),
                desc=f"COCO {sp} -> resized",
            ):
                if status == "ok":
                    ok += 1
                else:
                    fail += 1
                    tqdm.write(f"[FAIL] {image_id}: {msg}")

        print(f"COCO {sp} done. ok={ok} fail={fail}")

        if not args.keep_zips:
            try:
                zip_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()