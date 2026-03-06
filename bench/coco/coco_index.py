"""
COCO detection dataset, collate, and category mapping.

Wraps pycocotools for detection benchmarking.
"""

import os
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO


def build_category_mapping(coco_api: COCO) -> dict:
    """Map COCO category IDs to contiguous 0..79 indices."""
    cat_ids = sorted(coco_api.getCatIds())
    return {cat_id: i for i, cat_id in enumerate(cat_ids)}


class CocoDetectionDataset(Dataset):
    """COCO detection dataset returning (image_tensor, targets_dict)."""

    def __init__(self, root: str, ann_file: str, transform=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.getImgIds())
        self.transform = transform
        self.cat_map = build_category_mapping(self.coco)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            # Convert xywh → xyxy
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_map[ann["category_id"]])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        if self.transform is not None:
            img = self.transform(img)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return img, target


def det_collate_fn(batch):
    """Collate for variable-size detection targets."""
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    images = torch.stack(images, dim=0)
    return images, targets
