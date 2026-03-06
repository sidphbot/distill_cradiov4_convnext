"""Re-exports of metric functions from distill.model with clean public names."""

from distill.model import (
    _flatten_spatial_tokens as flatten_spatial_tokens,
    _cosine_sim as cosine_sim,
    _batch_retrieval_top1 as batch_retrieval_top1,
    _batch_retrieval_mrr as batch_retrieval_mrr,
    _centered_kernel_alignment as centered_kernel_alignment,
    _style_gram_loss as style_gram_loss,
    _spatial_energy as spatial_energy,
    _corrcoef_1d as corrcoef_1d,
    _topk_activation_f1 as topk_activation_f1,
    _pearson_corr_per_sample as pearson_corr_per_sample,
    _sobel_mag_2d as sobel_mag_2d,
    _laplacian_highpass_depthwise as laplacian_highpass_depthwise,
)

__all__ = [
    "flatten_spatial_tokens",
    "cosine_sim",
    "batch_retrieval_top1",
    "batch_retrieval_mrr",
    "centered_kernel_alignment",
    "style_gram_loss",
    "spatial_energy",
    "corrcoef_1d",
    "topk_activation_f1",
    "pearson_corr_per_sample",
    "sobel_mag_2d",
    "laplacian_highpass_depthwise",
]
