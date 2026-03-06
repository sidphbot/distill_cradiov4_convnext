"""Load teacher and student models for benchmarking."""

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPImageProcessor

from distill.model import DistillModel, SummaryHead, SpatialHead, teacher_forward_fixed
from bench.common.config import STUDENT_MODELS, DEFAULT_TEACHER_ID, DEFAULT_SIZE, DEFAULT_PATCH_SIZE


@dataclass
class TeacherBundle:
    model: nn.Module
    processor: CLIPImageProcessor
    device: str


@dataclass
class StudentBundle:
    model: DistillModel
    student_name: str
    size: int
    patch_size: int
    Ct: int
    Dt: int
    Ht: int
    Wt: int
    device: str


def load_teacher(teacher_id: str = DEFAULT_TEACHER_ID,
                 device: str = "cuda",
                 amp: bool = True) -> TeacherBundle:
    """Load frozen teacher model and processor."""
    processor = CLIPImageProcessor.from_pretrained(teacher_id)
    model = AutoModel.from_pretrained(teacher_id, trust_remote_code=True)
    model = model.to(device).eval()
    if amp and device.startswith("cuda"):
        model = model.half()
    for p in model.parameters():
        p.requires_grad_(False)
    return TeacherBundle(model=model, processor=processor, device=device)


def load_student(ckpt_path: str,
                 device: str = "cuda",
                 size: int = DEFAULT_SIZE,
                 patch_size: int = DEFAULT_PATCH_SIZE) -> StudentBundle:
    """Load student from checkpoint. Resolves timm name via STUDENT_MODELS mapping."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Resolve variant → timm name
    variant = ckpt.get("variant", "tiny")
    student_name = STUDENT_MODELS[variant]

    Ct = ckpt["Ct"]
    Dt = ckpt["Dt"]
    Ht = size // patch_size
    Wt = size // patch_size

    # Build student backbone
    student = timm.create_model(
        student_name,
        pretrained=False,
        features_only=True,
        out_indices=(2, 3),
    )

    # Infer channel dims from a dummy forward
    with torch.no_grad():
        f2, f3 = student(torch.randn(1, 3, size, size))
        Cs_sp = f2.shape[1]
        Cs = f3.shape[1]

    sum_head = SummaryHead(Cs, Ct)
    sp_head = SpatialHead(Cs_sp, Dt)

    spatial_norm_cap = ckpt.get("spatial_norm_cap", 50.0)
    model = DistillModel(
        student=student,
        sum_head=sum_head,
        sp_head=sp_head,
        Ht=Ht, Wt=Wt,
        spatial_norm_cap=spatial_norm_cap,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return StudentBundle(
        model=model,
        student_name=student_name,
        size=size,
        patch_size=patch_size,
        Ct=Ct, Dt=Dt, Ht=Ht, Wt=Wt,
        device=device,
    )
