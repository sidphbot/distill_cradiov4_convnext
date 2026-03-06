"""Shared constants and argparse helpers for benchmark scripts."""

import argparse

# Student model mapping (variant key → timm model name)
STUDENT_MODELS = {
    "tiny": "convnext_tiny.dinov3_lvd1689m",
    "small": "convnext_small.dinov3_lvd1689m",
}

DEFAULT_TEACHER_ID = "nvidia/C-RADIOv4-H"
DEFAULT_SIZE = 416
DEFAULT_PATCH_SIZE = 16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def add_teacher_args(parser: argparse.ArgumentParser):
    parser.add_argument("--teacher_id", type=str, default=DEFAULT_TEACHER_ID)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")


def add_student_args(parser: argparse.ArgumentParser):
    parser.add_argument("--student_ckpt", type=str, required=True,
                        help="Path to student checkpoint (.pt)")


def add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)


def add_output_args(parser: argparse.ArgumentParser):
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory for benchmark outputs")


def parse_amp(args) -> bool:
    """Resolve --amp / --no_amp flags."""
    if args.no_amp:
        return False
    return args.amp
