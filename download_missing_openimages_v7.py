#!/usr/bin/env python3
"""
Open Images V7 (mirror-only) downloader + resizer.

- Enumerates ImageIDs from official per-split CSVs (train/validation/test)
- Excludes any IDs already present in input_dir (by filename stem)
- Downloads pixels ONLY from the Open Images mirror bucket:
    s3://open-images-dataset/{split}/{image_id}.jpg
  using unsigned boto3 (no OriginalURL, no Flickr, no random hosts)
- ThreadPool for download (network-bound) + ProcessPool for resize (CPU-bound)
- Bounded in-flight pipeline to avoid RAM blowup

Output:
  output_dir/{image_id}.jpg

Install:
  pip install pillow tqdm boto3 botocore
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import threading
from pathlib import Path
from typing import Iterator, Tuple, Optional, Set

import boto3
import botocore
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from PIL import Image as PILImage
from tqdm import tqdm

from resize_and_save_images import _resize_save

# Official split ID lists (Open Images uses these for common downloadable splits).
# These are widely used "images-with-rotation"/"boxable-with-rotation" style files.
# If you have different desired lists, swap these URLs.
SPLIT_ID_CSVS = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}

BUCKET_NAME = "open-images-dataset"
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_TLS = threading.local()


def _get_bucket():
    """
    Thread-local unsigned bucket handle (connection reuse per thread).
    Mirrors the official downloader approach: unsigned access to open-images-dataset.
    """
    b = getattr(_TLS, "bucket", None)
    if b is None:
        s3 = boto3.resource(
            "s3",
            config=botocore.config.Config(signature_version=botocore.UNSIGNED),
        )
        b = s3.Bucket(BUCKET_NAME)
        _TLS.bucket = b
    return b


def load_exclude_ids(input_dir: Path) -> Set[str]:
    ids: Set[str] = set()
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in _EXTS:
            ids.add(p.stem)
    return ids


def iter_split_ids(csv_url: str) -> Iterator[str]:
    """
    Stream a split CSV and yield ImageID values.
    """
    import requests  # lazy (only used for the CSV fetch)
    with requests.get(csv_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        lines = (line.decode("utf-8", errors="replace") for line in r.iter_lines(chunk_size=1 << 20))
        reader = csv.DictReader(lines)
        for row in reader:
            try:
                image_id = row.get("ImageID") or row.get("ImageId") or row.get("image_id")
                if image_id:
                    yield image_id
            except Exception as e:
                print(f'Error reading Image {image_id}: str(e)')


def task_iter(exclude_ids: Set[str], output_dir: Path, overwrite: bool) -> Iterator[Tuple[str, str, str]]:
    """
    Yields (split, image_id, dst_path_str) for all ids in the split manifests,
    excluding those already present in input_dir and those already in output_dir (unless overwrite).
    """
    for split, csv_url in SPLIT_ID_CSVS.items():
        for image_id in iter_split_ids(csv_url):
            if image_id in exclude_ids:
                continue
            dst = output_dir / f"{image_id}.jpg"
            if dst.exists() and not overwrite:
                continue
            yield (split, image_id, str(dst))


def _download_one(task: Tuple[str, str, str], retries: int) -> Tuple[str, str, str, Optional[bytes], str]:
    """
    Download bytes from mirror bucket.
    Returns: (split, image_id, dst_str, img_bytes_or_none, err)
    """
    split, image_id, dst_str = task
    key = f"{split}/{image_id}.jpg"
    bucket = _get_bucket()

    last_err = ""
    for attempt in range(retries):
        try:
            buf = io.BytesIO()
            bucket.download_fileobj(key, buf)
            return (split, image_id, dst_str, buf.getvalue(), "")
        except botocore.exceptions.ClientError as e:
            # This includes NoSuchKey and other client errors
            last_err = str(e)
            # NoSuchKey won't succeed on retry; but keep 1 retry just in case of transient.
            if attempt == 0:
                continue
            return (split, image_id, dst_str, None, last_err)
        except Exception as e:
            last_err = str(e)
            # brief backoff
            import time
            time.sleep(min(10.0, 0.25 * (2 ** attempt)))

    return (split, image_id, dst_str, None, last_err)


def _resize_worker(
    image_id: str,
    img_bytes: bytes,
    dst_str: str,
    size: int,
    mode: str,
    quality: int,
) -> Tuple[str, str, str]:
    """
    Runs in a process. Returns (status, image_id, msg).
    """
    dst = Path(dst_str)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with PILImage.open(io.BytesIO(img_bytes)) as im:
            _resize_save(dst, im, mode, quality, size)
        return ("ok", image_id, "")
    except Exception as e:
        return ("fail", image_id, str(e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True, help="Existing {image_id}.jpg here will be EXCLUDED")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--mode", choices=["crop", "pad", "squash", "longest"], default="longest")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--overwrite", action="store_true")

    # Performance knobs
    ap.add_argument("--dl_workers", type=int, default=64, help="Download threads (network-bound)")
    ap.add_argument("--cpu_workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="Resize processes (CPU-bound)")
    ap.add_argument("--max_inflight", type=int, default=512, help="Max total in-flight tasks (bounds RAM)")
    ap.add_argument("--retries", type=int, default=3, help="Retries for mirror download (usually 2-3 is enough)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exclude_ids = load_exclude_ids(args.input_dir)
    print(f"Exclude IDs from input_dir: {len(exclude_ids):,}")
    print(f"Using mirror bucket only: s3://{BUCKET_NAME}/{{split}}/{{image_id}}.jpg")
    print(f"Split manifests: {', '.join(SPLIT_ID_CSVS.keys())}")

    tasks = task_iter(exclude_ids=exclude_ids, output_dir=args.output_dir, overwrite=args.overwrite)

    dl_futs = set()
    rs_futs = set()

    ok = miss = rs_fail = 0

    pbar = tqdm(desc="OpenImages mirror download+resize", unit="img")

    def drain_some(dl_ex, rs_ex):
        nonlocal dl_futs, rs_futs, ok, miss, rs_fail

        # Prefer draining resizes so CPU doesn't stall
        if rs_futs:
            done, rs_futs = wait(rs_futs, return_when=FIRST_COMPLETED)
            for f in done:
                status, image_id, msg = f.result()
                if status == "ok":
                    ok += 1
                else:
                    rs_fail += 1
                    tqdm.write(f"[RESIZE_FAIL] {image_id}: {msg}")
                pbar.update(1)
            return

        if dl_futs:
            done, dl_futs = wait(dl_futs, return_when=FIRST_COMPLETED)
            for f in done:
                split, image_id, dst_str, img_bytes, err = f.result()
                if img_bytes is None:
                    miss += 1
                    # Uncomment for visibility:
                    # tqdm.write(f"[MISS] {split}/{image_id}: {err}")
                    pbar.update(1)
                    continue
                rs_futs.add(
                    rs_ex.submit(
                        _resize_worker,
                        image_id,
                        img_bytes,
                        dst_str,
                        args.size,
                        args.mode,
                        args.quality,
                    )
                )
            return

    with ThreadPoolExecutor(max_workers=args.dl_workers) as dl_ex, ProcessPoolExecutor(max_workers=args.cpu_workers) as rs_ex:
        for t in tasks:
            # bound inflight (download + resize)
            while (len(dl_futs) + len(rs_futs)) >= args.max_inflight:
                drain_some(dl_ex, rs_ex)

            dl_futs.add(dl_ex.submit(_download_one, t, args.retries))

        # drain remaining
        while dl_futs or rs_futs:
            drain_some(dl_ex, rs_ex)

    pbar.close()
    print(f"Done. ok={ok} miss={miss} resize_fail={rs_fail}")


if __name__ == "__main__":
    main()