"""Muon optimizer: Newton-Schulz orthogonalization on gradients for matrix params.

Combined AdamW+Muon: Muon for ≥2D weight matrices, AdamW for everything else
(biases, norms, embeddings, 1D params).

Reference: https://github.com/KellerJordan/modded-nanogpt
"""

import torch
from torch.optim import Optimizer


def _newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the matrix sign function via 5-step Newton-Schulz iteration.

    Produces an orthogonal matrix that preserves the gradient direction
    while normalizing the spectrum — effectively a "spectral steepest descent" step.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon optimizer for matrix-shaped parameters.

    Applies Newton-Schulz orthogonalization to gradients before the update step,
    with optional momentum and weight decay.

    Args:
        params: Parameters to optimize (should be ≥2D weight matrices).
        lr: Learning rate.
        momentum: Momentum factor (SGD-style).
        nesterov: Use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iteration steps.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if g.ndim < 2:
                    # Fallback: plain SGD with momentum for non-matrix params
                    # (shouldn't happen if param groups are set up correctly)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    p.add_(g, alpha=-lr)
                    continue

                # For ≥2D params, reshape to 2D, apply Newton-Schulz, reshape back
                original_shape = g.shape
                if g.ndim > 2:
                    g = g.view(g.shape[0], -1)

                g = _newton_schulz5(g, steps=ns_steps)

                if len(original_shape) > 2:
                    g = g.view(original_shape)

                # Momentum
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                # Scale lr by max(1, fan_in/fan_out) for scale-invariance
                fan_in = p.shape[0]
                fan_out = p[0].numel()
                scale = max(1, fan_in / fan_out) ** 0.5
                p.add_(update, alpha=-lr * scale)

        return loss


def build_adamw_muon_optimizer(
    model: torch.nn.Module,
    lr_adamw: float,
    lr_muon: float,
    wd: float,
    betas: tuple,
    eps: float,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
):
    """Build a combined AdamW+Muon optimizer pair.

    - Muon handles ≥2D weight matrices (convolutions, linear layers)
    - AdamW handles biases, norms, and other 1D params

    Returns a list of optimizers: [adamw, muon] to be used with manual optimization.
    """
    adamw_decay_params = []
    adamw_no_decay_params = []
    muon_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1D params (norms, biases) → AdamW without decay
        if p.ndim == 1 or name.endswith(".bias"):
            adamw_no_decay_params.append(p)
        # ≥2D weight matrices → Muon
        elif p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_decay_params.append(p)

    optimizers = []

    # AdamW for non-matrix params
    adamw_groups = []
    if adamw_decay_params:
        adamw_groups.append({"params": adamw_decay_params, "weight_decay": wd})
    if adamw_no_decay_params:
        adamw_groups.append({"params": adamw_no_decay_params, "weight_decay": 0.0})
    if adamw_groups:
        adamw = torch.optim.AdamW(adamw_groups, lr=lr_adamw, betas=betas, eps=eps)
        optimizers.append(adamw)

    # Muon for matrix params
    if muon_params:
        muon = Muon(
            muon_params, lr=lr_muon,
            momentum=muon_momentum, nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
        )
        optimizers.append(muon)

    return optimizers
