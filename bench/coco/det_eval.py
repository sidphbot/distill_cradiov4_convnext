"""
COCO detection evaluation using pycocotools.

Usage:
    python -m bench.coco.det_eval \
        --det_ckpt /tmp/det/checkpoints/detector.pt \
        --student_ckpt checkpoints/best.pt \
        --coco_root /data/coco --out_dir /tmp/det_eval
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from bench.common.config import (
    add_output_args, DEFAULT_SIZE, IMAGENET_MEAN, IMAGENET_STD, parse_amp,
)
from bench.common.model_loaders import load_student
from bench.common.io import save_json, save_run_meta
from bench.coco.coco_index import CocoDetectionDataset, det_collate_fn, build_category_mapping
from bench.coco.det_train_headonly import (
    StudentBackboneWrapper, TeacherBackboneWrapper,
    build_detector, NUM_COCO_CLASSES,
)


def build_parser():
    parser = argparse.ArgumentParser(description="COCO detection evaluation")
    parser.add_argument("--det_ckpt", type=str, required=True,
                        help="Path to detector checkpoint")
    parser.add_argument("--student_ckpt", type=str, default="",
                        help="Path to student checkpoint (for student backbone)")
    parser.add_argument("--teacher_id", type=str, default="nvidia/C-RADIOv4-H")
    parser.add_argument("--coco_root", type=str, required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")
    add_output_args(parser)
    return parser


@torch.no_grad()
def evaluate_det(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_meta(str(out_dir), args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load detector checkpoint
    det_ckpt = torch.load(args.det_ckpt, map_location="cpu", weights_only=True)
    out_channels = det_ckpt["out_channels"]
    model_type = det_ckpt["model_type"]

    # Rebuild backbone
    if model_type == "student":
        student_b = load_student(args.student_ckpt, device, args.size, args.patch_size)
        backbone = StudentBackboneWrapper(student_b.model, out_channels)
    else:
        Ht = args.size // args.patch_size
        backbone = TeacherBackboneWrapper(out_channels, Ht, Ht)

    detector = build_detector(backbone, out_channels, NUM_COCO_CLASSES)
    detector.load_state_dict(det_ckpt["model_state_dict"])
    detector = detector.to(device).eval()

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(args.size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_ann_path = str(Path(args.coco_root) / "annotations" / "instances_val2017.json")
    val_img_dir = str(Path(args.coco_root) / "val2017")
    val_ds = CocoDetectionDataset(val_img_dir, val_ann_path, transform=transform)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=det_collate_fn, pin_memory=True,
    )

    # Build reverse category mapping (contiguous → COCO cat_id)
    coco_gt = COCO(val_ann_path)
    cat_ids = sorted(coco_gt.getCatIds())
    idx_to_cat = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    # Run inference
    coco_results = []
    for images, targets in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        outputs = detector(images)

        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"].item()
            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()

            for j in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[j].tolist()
                label_idx = labels[j].item()
                # Map back: label 0 = background in FasterRCNN, labels 1..80 = classes
                if label_idx == 0:
                    continue
                cat_id = idx_to_cat.get(label_idx - 1)
                if cat_id is None:
                    continue
                coco_results.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # xyxy → xywh
                    "score": scores[j].item(),
                })

    if not coco_results:
        print("No detections produced!")
        save_json({"mAP": 0, "AP50": 0, "AP75": 0, "APS": 0, "APM": 0, "APL": 0},
                  str(out_dir / "metrics.json"))
        return

    # Save predictions and run COCOeval
    pred_path = str(out_dir / "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "mAP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "APS": coco_eval.stats[3],
        "APM": coco_eval.stats[4],
        "APL": coco_eval.stats[5],
    }
    save_json(results, str(out_dir / "metrics.json"))
    print(f"\nmAP: {results['mAP']:.4f}, AP50: {results['AP50']:.4f}")
    print(f"Results saved to {out_dir / 'metrics.json'}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluate_det(args)


if __name__ == "__main__":
    main()
