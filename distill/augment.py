"""Student-only augmentations with ramped probability."""

import torch
import torchvision.transforms.functional as TF


def apply_student_augmentations(
    img_u8: torch.Tensor,
    p_global: float,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Apply lightweight augmentations to uint8 image batch before normalization.

    Args:
        img_u8: (B, 3, H, W) uint8 tensor
        p_global: master probability gating whether any aug is applied per sample
        rng: optional torch Generator for reproducibility

    Returns:
        (B, 3, H, W) float32 tensor in [0, 1] (augmented)
    """
    B = img_u8.shape[0]
    x = img_u8.float().div(255.0)  # (B, 3, H, W) float [0,1]

    if p_global <= 0.0:
        return x

    # Per-sample gate: with probability p_global, apply augmentations
    gate = torch.rand(B, generator=rng) < p_global  # (B,)

    for i in range(B):
        if not gate[i]:
            continue
        sample = x[i]  # (3, H, W)
        sample = _augment_single(sample, rng)
        x[i] = sample

    return x


def _augment_single(img: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
    """Apply individual augmentations to a single (3, H, W) float [0,1] image."""

    # Random brightness (±0.1), p=0.4
    if _coin(0.4, rng):
        factor = 1.0 + (torch.rand(1, generator=rng).item() * 0.2 - 0.1)
        img = (img * factor).clamp(0, 1)

    # Random contrast (±0.1), p=0.4
    if _coin(0.4, rng):
        factor = 1.0 + (torch.rand(1, generator=rng).item() * 0.2 - 0.1)
        mean = img.mean()
        img = ((img - mean) * factor + mean).clamp(0, 1)

    # Color jitter: hue ±0.02, saturation ±0.1, p=0.3
    if _coin(0.3, rng):
        hue_shift = torch.rand(1, generator=rng).item() * 0.04 - 0.02
        sat_factor = 1.0 + (torch.rand(1, generator=rng).item() * 0.2 - 0.1)
        img = TF.adjust_hue(img, hue_shift)
        img = TF.adjust_saturation(img, sat_factor)
        img = img.clamp(0, 1)

    # Gaussian blur (kernel 3 or 5, sigma 0.1-0.5), p=0.3
    if _coin(0.3, rng):
        kernel = 3 if torch.rand(1, generator=rng).item() < 0.5 else 5
        sigma = 0.1 + torch.rand(1, generator=rng).item() * 0.4
        img = TF.gaussian_blur(img.unsqueeze(0), kernel_size=kernel, sigma=sigma).squeeze(0)

    # ISO noise (additive Gaussian, std ~0.02), p=0.3
    if _coin(0.3, rng):
        std = 0.02
        noise = torch.randn_like(img) * std
        img = (img + noise).clamp(0, 1)

    return img


def _coin(p: float, rng: torch.Generator | None = None) -> bool:
    return torch.rand(1, generator=rng).item() < p
