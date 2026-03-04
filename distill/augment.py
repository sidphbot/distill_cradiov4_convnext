"""Student-only augmentations with ramped probability, powered by albumentations."""

import albumentations as A
import torch
from omegaconf import OmegaConf


def build_augmentation_pipeline(aug_cfg):
    """Deserialize the albumentations pipeline from config once at init."""
    return A.from_dict(OmegaConf.to_container(aug_cfg.pipeline, resolve=True))


def apply_student_augmentations(
    img_u8: torch.Tensor,
    p_global: float,
    pipeline: A.Compose,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Apply lightweight augmentations to uint8 image batch before normalization.

    Args:
        img_u8: (B, 3, H, W) uint8 tensor
        p_global: master probability gating whether any aug is applied per sample
        pipeline: pre-built albumentations Compose pipeline
        rng: optional torch Generator for reproducibility

    Returns:
        (B, 3, H, W) float32 tensor in [0, 1]
    """
    B = img_u8.shape[0]
    img_u8 = img_u8.cpu()

    if p_global <= 0.0:
        return img_u8.float().div(255.0)

    gate = torch.rand(B, generator=rng) < p_global

    out = torch.empty(B, 3, img_u8.shape[2], img_u8.shape[3], dtype=torch.float32)

    for i in range(B):
        if gate[i]:
            # torch (3,H,W) uint8 -> numpy (H,W,3) uint8
            np_img = img_u8[i].permute(1, 2, 0).numpy()
            augmented = pipeline(image=np_img)["image"]
            # numpy (H,W,3) uint8 -> torch (3,H,W) float [0,1]
            out[i] = torch.from_numpy(augmented).permute(2, 0, 1).float().div_(255.0)
        else:
            out[i] = img_u8[i].float().div_(255.0)

    return out
