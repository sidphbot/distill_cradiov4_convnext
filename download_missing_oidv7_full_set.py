#!/usr/bin/env python3
"""
Download Open Images "Full Set" excluding IDs present in input_dir,
resize, and save to output_dir as {image_id}.jpg.

This version is FAST:
- ThreadPoolExecutor for downloads (network-bound)
- ProcessPoolExecutor for resize/encode (CPU-bound)
- Bounded in-flight pipeline to avoid RAM blowup

Install:
  pip install pillow requests tqdm
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import time
import threading
from pathlib import Path
from typing import Iterator, Optional, Tuple

import requests
from PIL import Image as PILImage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED

from resize_and_save_images import _resize_save

FULLSET_CSV_URL = "https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv"

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# thread-local session (one session per thread => connection reuse)
_TLS = threading.local()


def _get_session() -> requests.Session:
    s = getattr(_TLS, "session", None)
    if s is None:
        s = requests.Session()
        _TLS.session = s
    return s


def load_exclude_ids(input_dir: Path) -> set[str]:
    ids = set()
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in _EXTS:
            ids.add(p.stem)
    return ids


def iter_fullset_rows(csv_path: Path) -> Iterator[dict]:
    """
    Read Open Images full-set CSV from local disk.
    Safe and fast (no network streaming).
    """
    with open(csv_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def pick_best_url(row: dict) -> Optional[str]:
    for key in ("OriginalURL", "Original Url", "URL", "ImageURL", "Thumbnail300KURL", "ThumbnailURL"):
        v = row.get(key)
        if v:
            return v
    for v in row.values():
        if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")):
            return v
    return None


def task_iter(
    csv_path: Path,
    exclude_ids: set[str],
    output_dir: Path,
    overwrite: bool,
) -> Iterator[Tuple[str, str, str]]:
    """
    Yields (image_id, url, dst_path_str)
    """
    for row in iter_fullset_rows(csv_path):
        image_id = row.get("ImageID") or row.get("ImageId") or row.get("image_id") or row.get("id")
        if not image_id:
            continue

        if image_id in exclude_ids:
            continue

        dst = output_dir / f"{image_id}.jpg"
        if dst.exists() and not overwrite:
            continue

        url = pick_best_url(row)
        if not url:
            continue

        yield (image_id, url, str(dst))


def _download_one(
    item: Tuple[str, str, str],
    timeout: float,
    retries: int,
) -> Tuple[str, Optional[bytes], str]:
    """
    Returns (image_id, bytes_or_none, err_msg)
    """
    image_id, url, _dst_str = item
    session = _get_session()

    last_err = ""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout, stream=True)

            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After", "")
                try:
                    sleep_s = float(ra)
                except Exception:
                    sleep_s = min(60.0, 0.5 * (2 ** attempt))
                time.sleep(sleep_s)
                last_err = "HTTP 429"
                continue

            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                time.sleep(min(60.0, 0.5 * (2 ** attempt)))
                continue

            return (image_id, resp.content, "")
        except Exception as e:
            last_err = str(e)
            time.sleep(min(60.0, 0.5 * (2 ** attempt)))

    return (image_id, None, last_err)


def _resize_worker(
    image_id: str,
    img_bytes: bytes,
    dst_str: str,
    size: int,
    mode: str,
    quality: int,
) -> Tuple[str, str]:
    """
    Runs in a process. Returns (status, msg)
    """
    dst = Path(dst_str)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with PILImage.open(io.BytesIO(img_bytes)) as im:
            # your helper handles modes + saving
            _resize_save(dst, im, mode, quality, size)
        return ("ok", "")
    except Exception as e:
        return ("fail", str(e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True, help="IDs here will be EXCLUDED from download")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--csv_path", type=Path, required=True,
                    help="Path to local image_ids_and_rotation.csv")

    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--mode", choices=["crop", "pad", "squash", "longest"], default="longest")
    ap.add_argument("--quality", type=int, default=95)

    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--overwrite", action="store_true")

    # NEW: separate pools
    ap.add_argument("--dl_workers", type=int, default=64, help="Download threads (network-bound)")
    ap.add_argument("--cpu_workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="Resize processes (CPU-bound)")
    ap.add_argument("--max_inflight", type=int, default=512, help="Max total in-flight (download+resize) tasks to bound RAM")
    ap.add_argument("--retries", type=int, default=6)

    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exclude_ids = load_exclude_ids(args.input_dir)
    print(f"Exclude IDs from input_dir: {len(exclude_ids):,}")

    tasks = task_iter(
        csv_path=args.csv_path,
        exclude_ids=exclude_ids,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    # In-flight future sets
    dl_futs = set()
    rs_futs = set()

    ok = fail = skip = dl_fail = 0

    # tqdm without a total (streaming)
    pbar = tqdm(desc="download+resize", unit="img")

    with ThreadPoolExecutor(max_workers=args.dl_workers) as dl_ex, ProcessPoolExecutor(max_workers=args.cpu_workers) as rs_ex:
        # Helper: submit download respecting max_inflight
        def maybe_submit_download(item):
            nonlocal dl_futs
            # Bound both pools combined
            while (len(dl_futs) + len(rs_futs)) >= args.max_inflight:
                _drain_some()
            dl_futs.add(dl_ex.submit(_download_one, item, args.timeout, args.retries))

        # Drain function: process whichever completes first
        def _drain_some():
            nonlocal dl_futs, rs_futs, ok, fail, dl_fail
            # If we have resize futures, prioritize draining them
            if rs_futs:
                done, rs_futs = wait(rs_futs, return_when=FIRST_COMPLETED)
                for f in done:
                    status, msg = f.result()
                    if status == "ok":
                        ok += 1
                    else:
                        fail += 1
                        tqdm.write(f"[RESIZE_FAIL] {msg}")
                    pbar.update(1)
                return

            if dl_futs:
                done, dl_futs = wait(dl_futs, return_when=FIRST_COMPLETED)
                for f in done:
                    image_id, img_bytes, err = f.result()
                    if img_bytes is None:
                        dl_fail += 1
                        # keep it quiet unless you want spam
                        # tqdm.write(f"[DL_FAIL] {image_id}: {err}")
                        pbar.update(1)
                        continue
                    # submit resize
                    # NOTE: destination known via closure? we need dst_str from task, so store it.
                return

        # We need dst_str mapping for downloaded results.
        # Easiest: wrap download item to include dst_str; and return it too.
        def _download_one_with_dst(item):
            image_id, url, dst_str = item
            image_id2, img_bytes, err = _download_one((image_id, url, dst_str), args.timeout, args.retries)
            return (image_id2, dst_str, img_bytes, err)

        # override dl submit to use wrapper
        def maybe_submit_download2(item):
            nonlocal dl_futs
            while (len(dl_futs) + len(rs_futs)) >= args.max_inflight:
                _drain_some2()
            dl_futs.add(dl_ex.submit(_download_one_with_dst, item))

        def _drain_some2():
            nonlocal dl_futs, rs_futs, ok, fail, dl_fail
            # Prefer draining resize futures first
            if rs_futs:
                done, rs_futs = wait(rs_futs, return_when=FIRST_COMPLETED)
                for f in done:
                    status, msg = f.result()
                    if status == "ok":
                        ok += 1
                    else:
                        fail += 1
                        tqdm.write(f"[RESIZE_FAIL] {msg}")
                    pbar.update(1)
                return

            if dl_futs:
                done, dl_futs = wait(dl_futs, return_when=FIRST_COMPLETED)
                for f in done:
                    image_id, dst_str, img_bytes, err = f.result()
                    if img_bytes is None:
                        dl_fail += 1
                        # tqdm.write(f"[DL_FAIL] {image_id}: {err}")
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

        # Pump tasks into pipeline
        for item in tasks:
            maybe_submit_download2(item)

        # Drain everything left
        while dl_futs or rs_futs:
            _drain_some2()

    pbar.close()
    print(f"Done. ok={ok} dl_fail={dl_fail} resize_fail={fail} (skipped handled upstream by excluding existing unless overwrite)")

if __name__ == "__main__":
    main()