"""
Train a linear probe (nn.Linear) on extracted frozen features.

Usage:
    python -m bench.imagenet.linear_probe \
        --train_features_dir /tmp/feat_train \
        --val_features_dir /tmp/feat_val \
        --epochs 50 --out_dir /tmp/probe
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from bench.common.config import add_output_args
from bench.common.io import save_json, save_run_meta


def build_parser():
    parser = argparse.ArgumentParser(description="Linear probe on extracted features")
    parser.add_argument("--train_features_dir", type=str, required=True)
    parser.add_argument("--val_features_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    add_output_args(parser)
    return parser


def load_sharded_features(features_dir: str):
    """Load and concatenate sharded .pt feature files."""
    shard_paths = sorted(Path(features_dir).glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {features_dir}")

    all_features = []
    all_labels = []
    for p in shard_paths:
        data = torch.load(p, map_location="cpu", weights_only=True)
        all_features.append(data["features"])
        all_labels.append(data["labels"])

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"Loaded {features.shape[0]} samples, dim={features.shape[1]} from {features_dir}")
    return features, labels


def accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res[f"top{k}"] = correct_k.item() / batch_size
        return res


def train_probe(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    save_run_meta(str(out_dir), args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load features
    train_feat, train_labels = load_sharded_features(args.train_features_dir)
    val_feat, val_labels = load_sharded_features(args.val_features_dir)

    D = train_feat.shape[1]
    num_classes = max(train_labels.max().item(), val_labels.max().item()) + 1
    print(f"Feature dim: {D}, num_classes: {num_classes}")

    # Dataloaders
    train_ds = TensorDataset(train_feat, train_labels)
    val_ds = TensorDataset(val_feat, val_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    probe = nn.Linear(D, num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_top1 = 0.0
    for epoch in range(args.epochs):
        # Train
        probe.train()
        for feat, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            feat, labels = feat.to(device), labels.to(device)
            logits = probe(feat)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Eval
        probe.eval()
        total_correct1 = 0
        total_correct5 = 0
        total_samples = 0
        with torch.no_grad():
            for feat, labels in val_loader:
                feat, labels = feat.to(device), labels.to(device)
                logits = probe(feat)
                acc = accuracy(logits, labels)
                bs = labels.size(0)
                total_correct1 += acc["top1"] * bs
                total_correct5 += acc["top5"] * bs
                total_samples += bs

        top1 = total_correct1 / total_samples
        top5 = total_correct5 / total_samples
        print(f"  Epoch {epoch+1}: top1={top1:.4f}, top5={top5:.4f}")

        if top1 > best_top1:
            best_top1 = top1
            torch.save(probe.state_dict(), ckpt_dir / "best_probe.pt")

    results = {
        "top1": best_top1,
        "top5": top5,
        "feature_dim": D,
        "num_classes": num_classes,
        "epochs": args.epochs,
    }
    save_json(results, str(out_dir / "metrics.json"))
    print(f"\nBest top-1: {best_top1:.4f}")
    print(f"Results saved to {out_dir / 'metrics.json'}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    train_probe(args)


if __name__ == "__main__":
    main()
