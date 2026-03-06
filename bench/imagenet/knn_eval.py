"""
Cosine kNN evaluation on extracted features.

Usage:
    python -m bench.imagenet.knn_eval \
        --train_features_dir /tmp/feat_train \
        --val_features_dir /tmp/feat_val \
        --k 20 --out_dir /tmp/knn
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from bench.common.config import add_output_args
from bench.common.io import save_json, save_run_meta
from bench.imagenet.linear_probe import load_sharded_features


def build_parser():
    parser = argparse.ArgumentParser(description="kNN evaluation on extracted features")
    parser.add_argument("--train_features_dir", type=str, required=True)
    parser.add_argument("--val_features_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    add_output_args(parser)
    return parser


@torch.no_grad()
def knn_eval(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_meta(str(out_dir), args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_feat, train_labels = load_sharded_features(args.train_features_dir)
    val_feat, val_labels = load_sharded_features(args.val_features_dir)

    num_classes = max(train_labels.max().item(), val_labels.max().item()) + 1
    k = args.k

    # L2 normalize
    train_feat = F.normalize(train_feat.float(), dim=-1).to(device)
    train_labels = train_labels.to(device)
    val_feat = F.normalize(val_feat.float(), dim=-1)
    val_labels = val_labels.to(device)

    # Chunked kNN (memory-friendly)
    chunk_size = 256
    total_correct1 = 0
    total_correct5 = 0
    total = 0

    for start in tqdm(range(0, val_feat.shape[0], chunk_size), desc="kNN"):
        end = min(start + chunk_size, val_feat.shape[0])
        query = val_feat[start:end].to(device)  # (chunk, D)
        gt = val_labels[start:end]               # (chunk,)

        # Cosine similarity
        sim = query @ train_feat.T  # (chunk, N_train)

        # Top-k
        topk_sim, topk_idx = sim.topk(k, dim=1)  # (chunk, k)
        topk_labels = train_labels[topk_idx]       # (chunk, k)

        # Weighted voting
        weights = (topk_sim / args.temperature).exp()  # (chunk, k)

        # Aggregate votes per class
        votes = torch.zeros(query.shape[0], num_classes, device=device)
        votes.scatter_add_(1, topk_labels, weights)

        # Top-1 and Top-5 predictions
        pred1 = votes.argmax(dim=1)
        total_correct1 += (pred1 == gt).sum().item()

        _, pred5 = votes.topk(5, dim=1)
        total_correct5 += pred5.eq(gt.unsqueeze(1)).any(dim=1).sum().item()

        total += gt.shape[0]

    top1 = total_correct1 / total
    top5 = total_correct5 / total

    results = {
        "top1": top1,
        "top5": top5,
        "k": k,
        "temperature": args.temperature,
        "num_train": train_feat.shape[0],
        "num_val": val_feat.shape[0],
    }
    save_json(results, str(out_dir / "metrics.json"))
    print(f"kNN top-1: {top1:.4f}, top-5: {top5:.4f}")
    print(f"Results saved to {out_dir / 'metrics.json'}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    knn_eval(args)


if __name__ == "__main__":
    main()
