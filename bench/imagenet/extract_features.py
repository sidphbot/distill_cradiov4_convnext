"""
Extract frozen features from teacher or student to disk.

Usage:
    python -m bench.imagenet.extract_features \
        --model student --student_ckpt checkpoints/best.pt \
        --split val --imagenet_root /data/imagenet \
        --feature f3_pool --out_dir /tmp/feat
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image

from distill.model import teacher_forward_fixed
from bench.common.config import (
    add_teacher_args, add_student_args, add_output_args,
    DEFAULT_SIZE, IMAGENET_MEAN, IMAGENET_STD, parse_amp,
)
from bench.common.model_loaders import load_teacher, load_student
from bench.common.preprocess import normalize_batch
from bench.common.io import save_run_meta


class FlatImageNetDataset(Dataset):
    """ImageNet stored as flat files with a JSON label map."""

    def __init__(self, img_dir: str, label_json: str, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        with open(label_json) as f:
            label_map = json.load(f)
        # Sort for deterministic order
        self.samples = sorted(label_map.items(), key=lambda x: x[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label = self.samples[idx]
        # Try common extensions
        for ext in (".jpg", ".jpeg", ".JPEG", ".png"):
            path = self.img_dir / f"{stem}{ext}"
            if path.exists():
                break
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_parser():
    parser = argparse.ArgumentParser(description="Extract frozen features to disk")
    parser.add_argument("--model", type=str, required=True, choices=["teacher", "student"])
    add_teacher_args(parser)
    parser.add_argument("--student_ckpt", type=str, default="")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--imagenet_root", type=str, required=True)
    parser.add_argument("--feature", type=str, default="f3_pool",
                        choices=["f3_pool", "summary"])
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    add_output_args(parser)
    return parser


class TeacherCollate:
    """Custom collate that keeps PIL images for teacher processor."""
    def __init__(self, size: int):
        self.size = size
        self.resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
        self.crop = transforms.CenterCrop(size)

    def __call__(self, batch):
        pil_imgs = []
        labels = []
        for img, label in batch:
            img = self.crop(self.resize(img))
            pil_imgs.append(img)
            labels.append(label)
        return pil_imgs, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def extract(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_meta(str(out_dir), args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = parse_amp(args) and device == "cuda"

    split_dir = Path(args.imagenet_root) / args.split
    ann_dir = Path(args.imagenet_root) / "annotations"
    label_json = ann_dir / f"{args.split}_labels.json"

    # Detect flat vs ImageFolder structure
    first_entry = next(split_dir.iterdir(), None)
    use_flat = label_json.exists() and first_entry is not None and first_entry.is_file()

    if args.model == "teacher":
        teacher_b = load_teacher(args.teacher_id, device, use_amp)

        if use_flat:
            ds = FlatImageNetDataset(str(split_dir), str(label_json))
        else:
            ds = datasets.ImageFolder(str(split_dir))
        collate = TeacherCollate(args.size)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
            pin_memory=True, drop_last=False,
        )

        all_features = []
        all_labels = []
        for pil_imgs, labels in tqdm(loader, desc="Extracting teacher features"):
            t_summary, t_spatial = teacher_forward_fixed(
                teacher_b.model, teacher_b.processor, pil_imgs,
                device, args.size, use_amp,
            )
            # Teacher always extracts summary
            all_features.append(t_summary.float().cpu())
            all_labels.append(labels)

            # Shard every 50k
            total = sum(f.shape[0] for f in all_features)
            if total >= 50000:
                _save_shard(all_features, all_labels, out_dir)
                all_features, all_labels = [], []

        if all_features:
            _save_shard(all_features, all_labels, out_dir)

    else:
        student_b = load_student(args.student_ckpt, device, args.size, args.patch_size)

        transform = transforms.Compose([
            transforms.Resize(args.size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        if use_flat:
            ds = FlatImageNetDataset(str(split_dir), str(label_json), transform=transform)
        else:
            ds = datasets.ImageFolder(str(split_dir), transform=transform)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )

        all_features = []
        all_labels = []
        for imgs, labels in tqdm(loader, desc="Extracting student features"):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                s_sum, s_tokens, s_sp, (f2, f3) = student_b.model(imgs)

            if args.feature == "f3_pool":
                feat = f3.mean(dim=(-2, -1))  # (B, Cs)
            else:  # summary
                feat = s_sum  # (B, Ct)

            all_features.append(feat.float().cpu())
            all_labels.append(labels)
            del imgs, s_sum, s_tokens, s_sp, f2, f3, feat

            total = sum(f.shape[0] for f in all_features)
            if total >= 50000:
                _save_shard(all_features, all_labels, out_dir)
                all_features, all_labels = [], []

        if all_features:
            _save_shard(all_features, all_labels, out_dir)

    print(f"Features saved to {out_dir}")


_shard_counter = 0

def _save_shard(features_list, labels_list, out_dir):
    global _shard_counter
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    path = out_dir / f"shard_{_shard_counter:04d}.pt"
    torch.save({"features": features, "labels": labels}, path)
    print(f"  Saved shard {_shard_counter}: {features.shape}")
    _shard_counter += 1


def main():
    global _shard_counter
    _shard_counter = 0
    parser = build_parser()
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
