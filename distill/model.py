import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F


# ============================================================
# MODEL WRAPPER (student + heads + distill ops)
# ============================================================

def build_param_groups(model: torch.nn.Module, weight_decay: float):
    """
    Exclude bias and norm parameters from weight decay.
    Common rule: no decay for 1D params (norm weights) and explicit biases.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class SummaryHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, x_bc):
        return self.ln2(self.fc(self.ln(x_bc)))


class SpatialHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        y = self.conv(x)  # (B,Dt,H,W)
        y = y.permute(0, 2, 3, 1)  # (B,H,W,Dt)
        y = self.ln(y)
        return y.permute(0, 3, 1, 2)  # (B,Dt,H,W)


def spatial_to_tokens(x_bdhw: torch.Tensor) -> torch.Tensor:
    # (B, D, H, W) -> (B, T, D)
    B, D, H, W = x_bdhw.shape
    return x_bdhw.permute(0, 2, 3, 1).reshape(B, H * W, D)


def l2norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine_loss(a, b):
    a = l2norm(a)
    b = l2norm(b)
    return (1.0 - (a * b).sum(dim=-1)).mean()


def cosine_loss_spatial_tokens(s_tokens, t_tokens):
    # both: (B, T, D)
    B, T, D = s_tokens.shape
    return cosine_loss(s_tokens.reshape(B * T, D), t_tokens.reshape(B * T, D))


def mse_loss(a, b):
    return F.mse_loss(a, b)

# ---- debug/metrics helpers (kept as-is, just grouped) ----

def _flatten_spatial_tokens(x_bdhw: torch.Tensor) -> torch.Tensor:
    # (B,D,H,W) -> (B,T,D)
    B, D, H, W = x_bdhw.shape
    return x_bdhw.permute(0, 2, 3, 1).reshape(B, H * W, D)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    a = a / (a.norm(dim=dim, keepdim=True) + eps)
    b = b / (b.norm(dim=dim, keepdim=True) + eps)
    return (a * b).sum(dim=dim)


def _corrcoef_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    return (x @ y) / ((x.norm() + eps) * (y.norm() + eps))


def _batch_retrieval_top1(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    sim = z_s @ z_t.T  # (B,B)
    nn = sim.argmax(dim=1)
    gt = torch.arange(z_s.shape[0], device=z_s.device)
    return (nn == gt).float().mean()


def _batch_retrieval_mrr(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    sim = z_s @ z_t.T  # (B,B)
    diag = sim.diag().unsqueeze(1)
    rank = 1 + (sim > diag).sum(dim=1)
    return (1.0 / rank.float()).mean()


def _gram_matrix(x_btd: torch.Tensor) -> torch.Tensor:
    B, T, D = x_btd.shape
    return torch.bmm(x_btd.transpose(1, 2), x_btd) / float(T)


def _style_gram_loss(s_btd: torch.Tensor, t_btd: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(_gram_matrix(s_btd), _gram_matrix(t_btd))


def _centered_kernel_alignment(z_s: torch.Tensor, z_t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    X = z_s - z_s.mean(dim=0, keepdim=True)
    Y = z_t - z_t.mean(dim=0, keepdim=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic = (K * L).sum()
    norm = torch.sqrt((K * K).sum() * (L * L).sum() + eps)
    return hsic / norm


def _spatial_energy(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., 1:] - x[..., :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    g = torch.sqrt(dx * dx + dy * dy + 1e-8)
    return g.mean(dim=(1, 2, 3))


def _to_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + eps)


@torch.no_grad()
def _normalize_per_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


@torch.no_grad()
def _make_side_by_side_channel_grid(
        s_bdhw: torch.Tensor,  # (D, H, W)
        t_bdhw: torch.Tensor,  # (D, H, W)
        max_px: int = 2000,
        max_channels: int = 256,
        padding: int = 2,
) -> torch.Tensor:
    from torchvision.utils import make_grid

    assert s_bdhw.ndim == 3 and t_bdhw.ndim == 3, f"expected (D,H,W), got {s_bdhw.shape}, {t_bdhw.shape}"

    D = min(int(s_bdhw.shape[0]), int(t_bdhw.shape[0]), int(max_channels))
    s0 = s_bdhw[:D].float()
    t0 = t_bdhw[:D].float()

    s_maps = _normalize_per_map(s0.unsqueeze(1))  # (D,1,H,W)
    t_maps = _normalize_per_map(t0.unsqueeze(1))

    nrow = max(1, int(math.floor(math.sqrt(D))))
    ncol = int(math.ceil(D / nrow))

    tile_w = max(1, (max_px - padding * (nrow - 1)) // nrow)
    tile_h = max(1, (max_px - padding * (ncol - 1)) // ncol)
    tile = int(max(8, min(64, tile_w, tile_h)))

    s_maps = F.interpolate(s_maps, size=(tile, tile), mode="bilinear", align_corners=False)
    t_maps = F.interpolate(t_maps, size=(tile, tile), mode="bilinear", align_corners=False)

    s_grid = make_grid(s_maps, nrow=nrow, padding=padding, normalize=False)  # (1,H,W)
    t_grid = make_grid(t_maps, nrow=nrow, padding=padding, normalize=False)

    if s_grid.shape[0] == 1:
        s_grid = s_grid.repeat(3, 1, 1)
    else:
        s_grid = s_grid[:1].repeat(3, 1, 1)

    if t_grid.shape[0] == 1:
        t_grid = t_grid.repeat(3, 1, 1)
    else:
        t_grid = t_grid[:1].repeat(3, 1, 1)

    Hs, Ws = int(s_grid.shape[1]), int(s_grid.shape[2])
    Ht, Wt = int(t_grid.shape[1]), int(t_grid.shape[2])
    H = max(Hs, Ht)
    if Hs != H:
        s_grid = F.pad(s_grid, (0, 0, 0, H - Hs), value=1.0)
    if Ht != H:
        t_grid = F.pad(t_grid, (0, 0, 0, H - Ht), value=1.0)

    sepW = padding * 4
    sep = torch.ones((3, H, sepW), dtype=s_grid.dtype, device=s_grid.device)

    out = torch.cat([s_grid, sep, t_grid], dim=2).clamp(0, 1)

    Hf, Wf = int(out.shape[1]), int(out.shape[2])
    scale = min(1.0, float(max_px) / float(max(Hf, Wf)))
    if scale < 1.0:
        newH = max(1, int(round(Hf * scale)))
        newW = max(1, int(round(Wf * scale)))
        out = F.interpolate(out.unsqueeze(0), size=(newH, newW), mode="bilinear", align_corners=False).squeeze(0)

    return out.clamp(0, 1)


@torch.no_grad()
def log_spatial_compare_first_sample(
        tb_writer,
        step: int,
        s_spatial_bdhw: torch.Tensor,  # (B, Dt, Ht, Wt)
        t_tokens_btd: torch.Tensor,  # (B, T, Dt)
        Ht: int,
        Wt: int,
        tag_prefix: str = "val/spatial_compare",
        max_side: int = 512,
        log_channel_grid: bool = True,
        channel_grid_max_px: int = 2000,
        channel_grid_max_channels: int = 256,
):
    # First sample
    s = s_spatial_bdhw[0].detach()  # (Dt, Ht, Wt)
    t_tokens = t_tokens_btd[0].detach()  # (T, Dt)
    assert t_tokens.shape[0] == Ht * Wt, f"T={t_tokens.shape[0]} != Ht*Wt={Ht * Wt}"

    # Teacher: (T,Dt) -> (Dt,Ht,Wt)
    t = t_tokens.transpose(0, 1).contiguous().reshape(-1, Ht, Wt)

    s_n = F.normalize(s, dim=0)
    t_n = F.normalize(t, dim=0)
    cos_map = (s_n * t_n).sum(dim=0)  # (Ht,Wt) in [-1,1]
    cos_map01 = (cos_map + 1.0) * 0.5

    l2_map = torch.sqrt(((s - t) ** 2).sum(dim=0) + 1e-8)
    l2_map01 = _to_01(l2_map)

    cos_img = cos_map01.unsqueeze(0).clamp(0, 1)
    l2_img = l2_map01.unsqueeze(0).clamp(0, 1)

    H, W = cos_img.shape[-2], cos_img.shape[-1]
    scale = min(1.0, float(max_side) / float(max(H, W)))
    if scale < 1.0:
        newH = max(1, int(round(H * scale)))
        newW = max(1, int(round(W * scale)))
        cos_img = F.interpolate(cos_img.unsqueeze(0), size=(newH, newW), mode="bilinear", align_corners=False).squeeze(
            0)
        l2_img = F.interpolate(l2_img.unsqueeze(0), size=(newH, newW), mode="bilinear", align_corners=False).squeeze(0)

    zero = torch.zeros_like(cos_img)
    rgb = torch.cat([cos_img, l2_img, zero], dim=0).clamp(0, 1)

    tb_writer.add_image(f"{tag_prefix}/cosine_map", cos_img, global_step=step)
    tb_writer.add_image(f"{tag_prefix}/l2_map", l2_img, global_step=step)
    tb_writer.add_image(f"{tag_prefix}/cos_l2_rgb", rgb, global_step=step)

    tb_writer.add_scalar(f"{tag_prefix}/cosine_mean", cos_map.mean().item(), step)
    tb_writer.add_scalar(f"{tag_prefix}/cosine_min", cos_map.min().item(), step)
    tb_writer.add_scalar(f"{tag_prefix}/cosine_max", cos_map.max().item(), step)
    tb_writer.add_scalar(f"{tag_prefix}/l2_mean", l2_map.mean().item(), step)
    tb_writer.add_scalar(f"{tag_prefix}/l2_p95", torch.quantile(l2_map.flatten(), 0.95).item(), step)

    if log_channel_grid:
        grid = _make_side_by_side_channel_grid(
            s_bdhw=s.detach().cpu(),
            t_bdhw=t.detach().cpu(),
            max_px=channel_grid_max_px,
            max_channels=channel_grid_max_channels,
            padding=2,
        )
        tb_writer.add_image(f"{tag_prefix}/side_by_side_grid", grid, global_step=step, dataformats="CHW")


class DistillModel(nn.Module):
    """
    Wraps:
      - timm student features_only backbone (returns f2,f3)
      - summary head (f3 pooled -> teacher summary dim Ct)
      - spatial head (f2 -> teacher spatial dim Dt)
    """

    def __init__(
            self,
            student: nn.Module,
            sum_head: nn.Module,
            sp_head: nn.Module,
            Ht: int,
            Wt: int,
            spatial_norm_cap: float = 50.0,
    ):
        super().__init__()
        self.student = student
        self.sum_head = sum_head
        self.sp_head = sp_head
        self.Ht = Ht
        self.Wt = Wt
        self.spatial_norm_cap = spatial_norm_cap

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          s_sum: (B,Ct)
          s_tokens: (B,T,Dt)
          s_sp: (B,Dt,Ht,Wt)
          (f2,f3) for optional debugging
        """
        f2, f3 = self.student(x)
        assert f2.shape[-2:] == (self.Ht, self.Wt), f"stage2 shape {f2.shape[-2:]} != {(self.Ht, self.Wt)}"

        s_sp = self.sp_head(f2)  # (B,Dt,H,W)
        # hotspot norm clamp (as per your investigation)
        norm = s_sp.norm(dim=1, keepdim=True).clamp_min(1e-6)
        scale = (self.spatial_norm_cap / norm).clamp_max(1.0)
        s_sp = s_sp * scale

        s_tokens = spatial_to_tokens(s_sp)  # (B,T,Dt)

        s_pool = f3.mean(dim=(-2, -1))
        s_sum = self.sum_head(s_pool)  # (B,Ct)
        return s_sum, s_tokens, s_sp, (f2, f3)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    # robust L1-like
    return torch.sqrt(x * x + eps * eps)


@torch.no_grad()
def make_sobel_kernels(device, dtype):
    # 3x3 Sobel kernels (normalized-ish)
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=device, dtype=dtype) / 8.0
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=device, dtype=dtype) / 8.0
    # shape for conv2d with groups=C: (C,1,3,3) will be created later
    return kx, ky


def sobel_grad_loss(
    s: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Edge-aware loss on feature maps using Sobel gradients.
    s,t: (B,C,H,W)
    """
    assert s.shape == t.shape and s.ndim == 4

    B, C, H, W = s.shape
    device, dtype = s.device, s.dtype

    kx, ky = make_sobel_kernels(device, dtype)
    kx = kx.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # (C,1,3,3)
    ky = ky.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    # depthwise conv
    sx = F.conv2d(s, kx, padding=1, groups=C)
    sy = F.conv2d(s, ky, padding=1, groups=C)
    tx = F.conv2d(t, kx, padding=1, groups=C)
    ty = F.conv2d(t, ky, padding=1, groups=C)

    # robust difference
    dx = charbonnier(sx - tx, eps=eps).mean()
    dy = charbonnier(sy - ty, eps=eps).mean()
    return dx + dy
