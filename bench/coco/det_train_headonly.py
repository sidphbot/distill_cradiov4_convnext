"""
Train FasterRCNN detection head on frozen student/teacher backbone.

Usage:
    python -m bench.coco.det_train_headonly \
        --model student --student_ckpt checkpoints/best.pt \
        --coco_root /data/coco --epochs 12 --out_dir /tmp/det
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from tqdm import tqdm

from bench.common.config import (
    add_teacher_args, add_student_args, add_output_args,
    DEFAULT_SIZE, IMAGENET_MEAN, IMAGENET_STD, parse_amp,
)
from bench.common.model_loaders import load_teacher, load_student
from bench.common.io import save_json, save_run_meta
from bench.coco.coco_index import CocoDetectionDataset, det_collate_fn


NUM_COCO_CLASSES = 80  # 80 foreground classes


class StudentBackboneWrapper(nn.Module):
    """Wraps DistillModel to return FPN-compatible feature dict from f2."""

    def __init__(self, distill_model, out_channels: int):
        super().__init__()
        self.distill_model = distill_model
        self.out_channels = out_channels

    def forward(self, x):
        with torch.no_grad():
            s_sum, s_tokens, s_sp, (f2, f3) = self.distill_model(x)
        return OrderedDict({"0": f2})


class TeacherBackboneWrapper(nn.Module):
    """Wraps pre-extracted teacher spatial features."""

    def __init__(self, out_channels: int, Ht: int, Wt: int):
        super().__init__()
        self.out_channels = out_channels
        self.Ht = Ht
        self.Wt = Wt

    def forward(self, x):
        # x is already the spatial features (B, D, H, W)
        return OrderedDict({"0": x})


def build_parser():
    parser = argparse.ArgumentParser(description="Head-only detection training")
    parser.add_argument("--model", type=str, required=True, choices=["teacher", "student"])
    add_teacher_args(parser)
    parser.add_argument("--student_ckpt", type=str, default="")
    parser.add_argument("--coco_root", type=str, required=True,
                        help="COCO root dir containing train2017/, val2017/, annotations/")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    add_output_args(parser)
    return parser


def build_detector(backbone_wrapper, out_channels, num_classes):
    """Build FasterRCNN with custom backbone and anchor generator."""
    # Single feature map → single tuple of anchor sizes
    anchor_sizes = ((32, 64, 128, 256),)
    aspect_ratios = ((0.5, 1.0, 2.0),)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    roi_pooler = torch.ops.torchvision.roi_align

    # Use FasterRCNN with the backbone wrapper
    backbone_wrapper.out_channels = out_channels
    detector = FasterRCNN(
        backbone=backbone_wrapper,
        num_classes=num_classes + 1,  # +1 for background
        rpn_anchor_generator=anchor_generator,
        min_size=200,
        max_size=600,
    )
    return detector


def train_det(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    save_run_meta(str(out_dir), args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = parse_amp(args) and device == "cuda"

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(args.size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ann = str(Path(args.coco_root) / "annotations" / "instances_train2017.json")
    train_img_dir = str(Path(args.coco_root) / "train2017")
    train_ds = CocoDetectionDataset(train_img_dir, train_ann, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=det_collate_fn, pin_memory=True,
    )

    # Build backbone + detector
    if args.model == "student":
        student_b = load_student(args.student_ckpt, device, args.size, args.patch_size)
        # Infer out_channels from f2
        with torch.no_grad():
            dummy = torch.randn(1, 3, args.size, args.size, device=device)
            _, _, _, (f2_dummy, _) = student_b.model(dummy)
            out_channels = f2_dummy.shape[1]
            del dummy, f2_dummy

        backbone = StudentBackboneWrapper(student_b.model, out_channels)
        for p in backbone.distill_model.parameters():
            p.requires_grad_(False)
    else:
        teacher_b = load_teacher(args.teacher_id, device, use_amp)
        # For teacher, we extract spatial features and reshape
        # Infer Dt from a forward pass
        from distill.model import teacher_forward_fixed
        from bench.common.preprocess import pad_resize
        dummy_pil = [Image.new("RGB", (args.size, args.size))]
        import PIL.Image as Image
        t_sum, t_tok = teacher_forward_fixed(
            teacher_b.model, teacher_b.processor, dummy_pil,
            device, args.size, use_amp,
        )
        Dt = t_tok.shape[-1]
        Ht = args.size // args.patch_size
        out_channels = Dt
        del t_sum, t_tok

        backbone = TeacherBackboneWrapper(out_channels, Ht, Ht)

    detector = build_detector(backbone, out_channels, NUM_COCO_CLASSES)
    detector = detector.to(device)

    # Freeze backbone parameters, only train RPN + ROI heads
    for name, p in detector.named_parameters():
        if "backbone" in name:
            p.requires_grad_(False)

    trainable_params = [p for p in detector.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs * 2 // 3, gamma=0.1)

    # Training loop
    for epoch in range(args.epochs):
        detector.train()
        total_loss = 0.0
        n_batches = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = detector(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            n_batches += 1

        lr_scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    # Save detector checkpoint
    torch.save({
        "model_state_dict": detector.state_dict(),
        "model_type": args.model,
        "out_channels": out_channels,
        "num_classes": NUM_COCO_CLASSES,
        "size": args.size,
        "patch_size": args.patch_size,
    }, ckpt_dir / "detector.pt")
    print(f"Detector saved to {ckpt_dir / 'detector.pt'}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    train_det(args)


if __name__ == "__main__":
    main()
