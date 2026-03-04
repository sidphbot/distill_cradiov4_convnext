"""Student-only augmentations with ramped probability, powered by albumentations."""

import albumentations as A
import torch
from omegaconf import OmegaConf


def build_augmentation_pipeline(aug_cfg):
    """Deserialize the albumentations pipeline from config once at init."""
    return A.from_dict(OmegaConf.to_container(aug_cfg.pipeline, resolve=True))


def apply_student_augmentations(
    img_u8: torch.Tensor,
    strength: float,
    pipeline: A.Compose,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Apply augmentations to uint8 image batch, blended by strength.

    Every sample is always augmented through the pipeline. The final output
    is an alpha-blend between clean and augmented: ``(1-s)*clean + s*aug``.
    This keeps consistency loss informative at all ramp stages.

    Args:
        img_u8: (B, 3, H, W) uint8 tensor
        strength: blend factor in [0, 1] — 0 = clean, 1 = fully augmented
        pipeline: pre-built albumentations Compose pipeline
        rng: optional torch Generator for reproducibility

    Returns:
        (B, 3, H, W) float32 tensor in [0, 1]
    """
    B = img_u8.shape[0]
    img_u8 = img_u8.cpu()

    clean = img_u8.float().div_(255.0)

    if strength <= 0.0:
        return clean

    aug = torch.empty_like(clean)
    for i in range(B):
        np_img = img_u8[i].permute(1, 2, 0).numpy()
        augmented = pipeline(image=np_img)["image"]
        aug[i] = torch.from_numpy(augmented).permute(2, 0, 1).float().div_(255.0)

    if strength >= 1.0:
        return aug

    return torch.lerp(clean, aug, strength)
