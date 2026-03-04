"""Optuna hyperparameter tuning for distillation.

Usage:
    python -m distill.tune --config distill/config.yaml --tune-config distill/tune_config.yaml [dotlist overrides]

Dotlist overrides apply to the base training config (same as launcher).
Tune-specific settings (search space, trial overrides, study params) come from --tune-config.
"""

import argparse
import copy
import math
import time
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModel, CLIPImageProcessor

from distill.data import DistillDataModule
from distill.model import teacher_forward_fixed
from distill.lightning_module import DistillLightningModule
from distill.launcher import build_model, _generate_data_list


class OptunaCallback(pl.Callback):
    """Reports alignment_score to Optuna and prunes bad trials."""

    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_train_epoch_end(self, trainer, pl_module):
        score = pl_module._last_alignment_score
        self.trial.report(score, trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# Shared teacher state (loaded once, reused across trials)
_teacher_state = {}


def _get_teacher(cfg, device):
    """Load teacher once and cache it."""
    if "teacher" not in _teacher_state:
        teacher_proc = CLIPImageProcessor.from_pretrained(cfg.teacher.id)
        teacher = AutoModel.from_pretrained(
            cfg.teacher.id, trust_remote_code=True
        ).to(device).eval().half()
        for p in teacher.parameters():
            p.requires_grad_(False)
        _teacher_state["teacher"] = teacher
        _teacher_state["teacher_proc"] = teacher_proc
    return _teacher_state["teacher"], _teacher_state["teacher_proc"]


def _infer_teacher_dims(teacher, teacher_proc, cfg, device):
    """Get teacher output dimensions from a dummy forward pass."""
    if "dims" in _teacher_state:
        return _teacher_state["dims"]

    from PIL import Image
    dummy = [Image.new("RGB", (cfg.data.size, cfg.data.size))]
    t_sum, t_tok = teacher_forward_fixed(
        teacher=teacher, proc=teacher_proc, pil_imgs=dummy,
        device=device, size=cfg.data.size, amp=cfg.teacher.amp,
    )
    Ct = int(t_sum.shape[-1])
    Dt = int(t_tok.shape[-1])
    Ht = cfg.data.size // cfg.data.patch_size
    Wt = cfg.data.size // cfg.data.patch_size
    del t_sum, t_tok
    torch.cuda.empty_cache()
    dims = (Ct, Dt, Ht, Wt)
    _teacher_state["dims"] = dims
    return dims


def _sample_search_space(trial, search_space):
    """Sample all params from the search space config. Returns {dotpath: value}."""
    sampled = {}
    for dotpath, spec in search_space.items():
        # Use dotpath with . replaced as trial param name for readability
        name = dotpath.replace(".", "_")
        typ = spec["type"]
        if typ == "log_float":
            sampled[dotpath] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif typ == "float":
            sampled[dotpath] = trial.suggest_float(name, spec["low"], spec["high"])
        elif typ == "int":
            sampled[dotpath] = trial.suggest_int(name, spec["low"], spec["high"])
        elif typ == "categorical":
            sampled[dotpath] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unknown search space type '{typ}' for {dotpath}")
    return sampled


def objective(trial, base_cfg, tune_cfg):
    """Single Optuna trial: sample HPs, train for a few epochs, return alignment_score."""
    cfg = copy.deepcopy(base_cfg)

    # 1. Apply trial_overrides from tune config (data caps, epochs, etc.)
    if "trial_overrides" in tune_cfg:
        cfg = OmegaConf.merge(cfg, tune_cfg.trial_overrides)

    # 2. Sample hyperparameters from search space and apply to cfg
    sampled = _sample_search_space(trial, tune_cfg.search_space)
    for dotpath, value in sampled.items():
        OmegaConf.update(cfg, dotpath, value)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. Load teacher (cached across trials)
    teacher, teacher_proc = _get_teacher(cfg, device)
    Ct, Dt, Ht, Wt = _infer_teacher_dims(teacher, teacher_proc, cfg, device)

    # 4. Build data module
    dm = DistillDataModule(cfg)
    dm.setup()

    # 5. Build student model (fresh each trial)
    student_name = cfg.model.student_models[cfg.model.student_variant]
    model = build_model(cfg, device, Ct, Dt, Ht, Wt, student_name)

    steps_per_epoch = math.ceil(len(dm.train_ds) / cfg.dataloader.batch_size)

    # 6. Create Lightning module
    trial_name = f"trial_{trial.number}_{time.strftime('%Y%m%d%H%M%S')}"
    lit_module = DistillLightningModule(
        cfg=cfg,
        model=model,
        teacher=teacher,
        teacher_proc=teacher_proc,
        Ct=Ct, Dt=Dt, Ht=Ht, Wt=Wt,
        steps_per_epoch=steps_per_epoch,
    )
    lit_module.checkpoint_dir = Path(cfg.experiment.root) / "optuna" / trial_name / "checkpoints"

    tb_logger = TensorBoardLogger(
        save_dir=str(Path(cfg.experiment.root) / "optuna" / trial_name),
        name="tb",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=tb_logger,
        val_check_interval=cfg.logging.eval_every * cfg.training.grad_accum,
        limit_val_batches=cfg.logging.eval_batches,
        log_every_n_steps=50,
        enable_progress_bar=True,
        callbacks=[OptunaCallback(trial)],
    )

    # 7. Train
    trainer.fit(lit_module, datamodule=dm)

    score = lit_module._last_alignment_score

    # 8. Cleanup GPU memory
    del model, lit_module, trainer
    torch.cuda.empty_cache()

    return score


def main():
    ap = argparse.ArgumentParser(description="Optuna HP tuning for distillation")
    ap.add_argument("--config", type=str, default="distill/config.yaml",
                    help="Path to base training YAML config")
    ap.add_argument("--tune-config", type=str, default="distill/tune_config.yaml",
                    help="Path to tune YAML config (search space, trial overrides, study params)")
    args, overrides = ap.parse_known_args()

    # Load base config + CLI dotlist overrides (same as launcher)
    base_cfg = OmegaConf.load(args.config)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    # Handle image_dir -> data_list generation (same as launcher)
    image_dir = base_cfg.data.get("image_dir", "")
    if image_dir and not base_cfg.data.data_list:
        base_cfg.data.data_list = _generate_data_list(image_dir)
        base_cfg.data.image_dir = ""
    elif image_dir and base_cfg.data.data_list:
        raise SystemExit("Provide data.data_list or data.image_dir, not both.")

    if not base_cfg.data.data_list:
        raise SystemExit("data.data_list or data.image_dir is required")

    # Load tune config
    tune_cfg = OmegaConf.load(args.tune_config)
    study_cfg = tune_cfg.study

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=study_cfg.seed, n_startup_trials=study_cfg.n_startup),
        pruner=MedianPruner(
            n_warmup_steps=study_cfg.pruner_warmup_steps,
            n_startup_trials=study_cfg.pruner_startup_trials,
        ),
        storage=study_cfg.storage,
        study_name=study_cfg.name,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, base_cfg, tune_cfg),
        n_trials=study_cfg.n_trials,
    )

    # Print results
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    best = study.best_trial
    print(f"  Score (alignment_score): {best.value:.4f}")
    print(f"  Trial number: {best.number}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Build CLI override string using original dotpaths from search space
    print("\nRun full training with:")
    cli_parts = []
    for dotpath in tune_cfg.search_space:
        name = dotpath.replace(".", "_")
        if name in best.params:
            cli_parts.append(f"{dotpath}={best.params[name]}")
    print(f"  python -m distill.launcher --config {args.config} {' '.join(cli_parts)}")


if __name__ == "__main__":
    main()
