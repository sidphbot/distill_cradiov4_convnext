#!/usr/bin/env python3
import os
import argparse
import random
from pathlib import Path

import fiftyone.zoo as foz
import fiftyone.core.media as fom
from tqdm import tqdm


def load_cached_ids(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.add(s)
    return out


def sanitize(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").strip()


def is_image_ok(path: str, deep_verify: bool = False) -> bool:
    if not path or not os.path.isfile(path):
        return False

    # fast media-type check
    try:
        if fom.get_media_type(path) != "image":
            return False
    except Exception:
        return False

    if deep_verify:
        # slower but catches truncated/corrupt JPEGs
        try:
            from PIL import Image
            with Image.open(path) as im:
                im.verify()
        except Exception:
            return False

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append_n", type=int, required=True)
    ap.add_argument("--out", type=str, required=True, help="candidate list file to append to (image_ids.txt)")
    ap.add_argument("--cached_ids_file", type=str, default=None, help="cached ids log to avoid duplicates")
    ap.add_argument(
        "--pool_max_samples",
        type=int,
        default=None,
        help="how many local image files to consider as a pool (None = all found). "
             "If you keep hitting duplicates, raise this.",
    )
    ap.add_argument(
        "--deep_verify",
        action="store_true",
        help="Use PIL verify() to skip truncated/corrupt images (slower).",
    )
    ap.add_argument(
        "--ext",
        type=str,
        default=".jpg",
        help="File extension to scan for (default: .jpg). Use '*' to scan all files.",
    )
    args = ap.parse_args()

    cached = load_cached_ids(args.cached_ids_file) if args.cached_ids_file else set()

    # Ensure dataset is present locally (safe if already downloaded)
    foz.download_zoo_dataset("open-images-v6", split=args.split)

    # Locate the zoo dataset directory and the split data directory
    zoo_dir = foz.find_zoo_dataset("open-images-v6", split=args.split)
    data_dir = os.path.join(zoo_dir, args.split, "data")
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Could not find Open Images v6 data dir: {data_dir}")

    # Collect candidate filepaths
    ext = args.ext.lower()
    if ext == "*" or ext == ".*":
        paths = [str(p) for p in Path(data_dir).iterdir() if p.is_file()]
    else:
        if not ext.startswith("."):
            ext = "." + ext
        paths = [str(p) for p in Path(data_dir).glob(f"*{ext}")]

    if not paths:
        raise RuntimeError(f"No files found in {data_dir} with ext={args.ext}")

    # Deterministic shuffle
    rng = random.Random(args.seed)
    rng.shuffle(paths)

    # Apply pool limit
    if args.pool_max_samples is not None:
        paths = paths[: args.pool_max_samples]

    appended = 0
    scanned = 0
    skipped_bad = 0
    skipped_dup = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "a", encoding="utf-8") as f:
        for p in paths:
            scanned += 1

            if not is_image_ok(p, deep_verify=args.deep_verify):
                skipped_bad += 1
                continue

            image_id = sanitize(Path(p).stem)

            if image_id in cached:
                skipped_dup += 1
                continue

            f.write(image_id + "\n")
            cached.add(image_id)
            appended += 1

            if appended >= args.append_n:
                break

    print(
        f"Scanned {scanned} files from pool; appended {appended} new IDs to {args.out} | "
        f"skipped_bad={skipped_bad}, skipped_dup={skipped_dup}"
    )
    if appended < args.append_n:
        print(
            "WARNING: pool exhausted before reaching append_n. "
            "Increase --pool_max_samples, or ensure your cached_ids_file isn't too restrictive."
        )


if __name__ == "__main__":
    # main()
    data_root = Path('/home/burplord/fiftyone/open-images-v6/train/data')
    append_n = 30000
    out_file = Path('/home/burplord/data/openimages_cradiov4_teacher_outputs/image_ids.txt')

    image_ids = [p.stem for p in data_root.glob('*.jpg')]
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    cached = set(out_file.read_text().split('\n'))
    skipped_dup = 0
    appended = 0

    with open(out_file, "a", encoding="utf-8") as f:
        for image_id in tqdm(image_ids):

            if image_id in cached:
                skipped_dup += 1
                continue

            f.write(image_id + "\n")
            cached.add(image_id)
            appended += 1

            if appended >= append_n:
                break