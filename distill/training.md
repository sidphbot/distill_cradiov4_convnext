# Distillation Training Workflow

## Quick Start

```bash
# Full training (defaults from config.yaml)
python -m distill.launcher --config distill/config.yaml data.data_list=/path/to/list.txt

# From an image directory (auto-generates data_list.txt)
python -m distill.launcher --config distill/config.yaml data.image_dir=/path/to/images/

# Debug mode (applies debug overlay: reduced data caps + frequent eval)
python -m distill.launcher --config distill/config.yaml data.data_list=/path/to/list.txt mode=debug

# CLI overrides (any dotpath into config.yaml)
python -m distill.launcher --config distill/config.yaml data.data_list=/path/to/list.txt \
    dataloader.batch_size=16 training.lr=3e-4 training.epochs=10
```

All parameters live in `distill/config.yaml`. CLI arguments are OmegaConf dotlist overrides merged on top. When `mode=debug`, the `debug:` section in the YAML is merged as an overlay (caps data, increases eval frequency) and `_debug` is appended to the experiment suffix.

## Architecture

- **Teacher**: C-RADIOv4-H (ViT-Huge), frozen, fp16. Produces summary embedding `(B, Ct)` and spatial tokens `(B, T, Dt)`. Model ID set by `teacher.id`.
- **Student**: ConvNeXt variant selected by `model.student_variant` (mapped to a timm model name via `model.student_models`), `features_only` with stages 2+3. Stage 3 pooled → `SummaryHead → (B, Ct)`. Stage 2 → `SpatialHead → (B, Dt, Ht, Wt)`.
- Both operate at `data.size` × `data.size` input, `data.patch_size` determines spatial grid: `Ht = Wt = size / patch_size`, `T = Ht * Wt`.

## Configuration

All training parameters are in `distill/config.yaml`. Key sections:

| Section | Controls |
|---|---|
| `model` | Student variant, model names, spatial norm cap, grad checkpointing |
| `teacher` | Teacher model ID, teacher AMP |
| `data` | Data list path, image dir, val fraction, input size, patch size, data caps, normalization stats |
| `dataloader` | Batch size, num workers, persistent workers, prefetch factor |
| `training` | Epochs, LR, weight decay, AMP, grad accumulation, grad clip, optimizer betas/eps |
| `loss` | Lambda weights (distillation + consistency), grad eps, ramp schedules (start/end/warmup_frac), steps per epoch |
| `augmentation` | Warmup steps, probability ramp (p_start → p_end), albumentations pipeline definition |
| `logging` | Log frequency, eval frequency/batches, image grid settings, spatial compare settings, mem tracking interval, quantile list |
| `experiment` | Output root directory, experiment suffix |
| `debug` | Overlay merged when `mode=debug` (overrides data caps and logging frequency) |

## Data Pipeline

1. `ImagePathDataset` returns raw file paths; `OnlineCollate` loads + resizes at batch time.
2. **Resize**: 50/50 deterministic (hash of `seed:path`) choice between aspect-ratio-preserving pad (black canvas) and squash to `data.size` × `data.size`. Seed changes each epoch for variation.
3. Collate returns `(img_u8, pil_rs, keys)` — uint8 tensor for student, PIL list for teacher processor, paths for debugging.
4. Train/val split is deterministic via stable hash on path, controlled by `data.val_frac`.

## Training Step (Manual Optimization)

Each call to `training_step` is one micro-batch with a **dual student pass**:

1. **Teacher forward** (`@torch.no_grad`, fp16 autocast): teacher processor → teacher model → `(t_summary, t_spatial_tokens)`.
2. **Ramp loss weights** (see Ramps below).
3. **Prepare two inputs on CPU**:
   - `x_clean`: `img_u8.float()/255 → normalize` (no augmentation)
   - `x_aug`: `apply_student_augmentations → normalize`
4. **Clean student forward** under fp16 autocast → `(s_sum_c, s_tokens_c, s_sp_c, (f2, f3))` — used for distillation losses against teacher.
5. **Distillation loss**: `compute_losses(s_sum_c, s_tokens_c, s_sp_c, ...)` — same as before, on clean outputs.
6. **Augmented student forward** under fp16 autocast → `(s_sum_a, s_tokens_a, s_sp_a, _)`.
7. **Consistency loss** (augmentation invariance regularizer):
   - `cons_summary = cosine_loss(s_sum_a, s_sum_c.detach())`
   - `cons_spatial = cosine_loss_spatial_tokens(s_tokens_a, s_tokens_c.detach())`
   - Clean outputs are `.detach()`'d so gradients only push the augmented path to match clean, not vice versa.
8. **Total loss**: `loss = distill_loss + λ_cs * cons_summary + λ_csp * cons_spatial`.
9. **Extract log scalars** as Python floats before backward.
10. **`scaler.scale(loss / grad_accum).backward()`** — graph freed immediately.
11. **Delete all GPU intermediates**.
12. Every `training.grad_accum` micro-steps: unscale → `clip_grad_norm(training.grad_clip)` → scaler.step → scaler.update → `zero_grad(set_to_none=True)`.

## Loss Function

```
distill_loss = λ_summary * loss_sum + λ_spatial * loss_sp
loss_sum = cosine_loss(s_sum_c, t_sum) + λ_mse * mse(s_sum_c, t_sum)
loss_sp  = cosine_loss(s_tokens_c, t_tokens) + λ_sp_mse * mse(s_sp_c, t_sp) + λ_grad * sobel_grad_loss(s_sp_c, t_sp)

cons_summary = cosine_loss(s_sum_a, s_sum_c.detach())
cons_spatial = cosine_loss_spatial_tokens(s_tokens_a, s_tokens_c.detach())
loss_consistency = λ_cs * cons_summary + λ_csp * cons_spatial

total_loss = distill_loss + loss_consistency
```

- `cosine_loss`: `(1 - cos_sim).mean()` on L2-normalized vectors.
- `sobel_grad_loss`: Sobel edge maps on both student/teacher spatial feature maps, Charbonnier-robust L1 on the difference (eps from `loss.grad_eps`).
- Spatial norm capping at `model.spatial_norm_cap` on student output to prevent hotspots.
- **Consistency loss**: enforces augmentation invariance by pushing augmented student outputs to match clean student outputs (detached). Separate lambdas for summary and spatial.

Weights `λ_summary`, `λ_spatial`, `λ_mse` come from `loss.lambda_summary`, `loss.lambda_spatial`, `loss.lambda_mse`. All dynamic weights (`λ_sp_mse`, `λ_grad`, `λ_cs`, `λ_csp`) are ramped (see below).

## Ramps

All use `ramp_linear(step, warmup_steps, start, end)` — linear interpolation clamped at end. Configured via `loss.ramp` and `augmentation`:

| Weight | Config path | Purpose |
|---|---|---|
| `λ_sp_mse` | `loss.ramp.mse_sp.{start,end,warmup_frac}` | Ease spatial MSE pressure early |
| `λ_grad` | `loss.ramp.grad.{start,end,warmup_frac}` | Let coarse structure form before enforcing edges |
| `λ_cs` | `loss.ramp.cons_summary.{start,end,warmup_frac}` | Summary consistency weight (ramps 2x faster than spatial) |
| `λ_csp` | `loss.ramp.cons_spatial.{start,end,warmup_frac}` | Spatial consistency weight |
| `aug_strength` | `augmentation.{warmup_frac,strength_start,strength_end}` | Ramp augmentation intensity — alpha-blend between clean and augmented |

All `warmup_frac` values are fractions of the first epoch's step count (auto-computed from `len(train_dataset) / batch_size`).

## Augmentations (Student Only)

Augmentations use **albumentations** and are defined declaratively in `augmentation.pipeline` in the config YAML. The pipeline is deserialized once at init via `A.from_dict()` and reused every step.

Every sample is always augmented through the pipeline. The output is alpha-blended between clean and augmented: `(1-s)*clean + s*aug`, where `s` (strength) ramps from `augmentation.strength_start` → `augmentation.strength_end` over `augmentation.warmup_frac` of the first epoch. This ensures consistency loss always has gradient signal while smoothly increasing augmentation intensity.

Teacher always sees clean resized images (no augmentations).

To change augmentations, edit the `augmentation.pipeline` section in `config.yaml`. The format follows `albumentations.from_dict()` / `to_dict()` conventions — each transform has a `__class_fullname__` and its parameters.

## Validation

- Runs every `logging.eval_every * training.grad_accum` training batches.
- `logging.eval_batches` caps how many val batches per run.
- `validation_step` is wrapped in `@torch.no_grad()`. Forward + loss on GPU, then everything moved to CPU for diagnostics. GPU freed + `empty_cache()` after each val epoch.
- **Accumulators persist across all val runs within a training epoch** — results are only reported once in `on_train_epoch_end`.

## Logging

### Train (TensorBoard, every `logging.log_every` steps via Lightning)
- `train/loss`, `train/loss_sum`, `train/loss_sp` (weighted compound losses)
- `train/cos_sum`, `train/cos_sp`, `train/mse_sum`, `train/mse_sp`, `train/grad` (raw loss components)
- `train/cons_summary`, `train/cons_spatial`, `train/loss_cons` (consistency loss terms)
- `train/w/mse_sp`, `train/w/grad`, `train/w/cons_summary`, `train/w/cons_spatial` (current ramp weights)
- `train/aug_strength` (current augmentation strength)

### Train images (every `logging.log_every * logging.image_log_multiplier` steps)
- `train/teacher_input`: grid of un-augmented inputs (`logging.image_grid.n` images, `logging.image_grid.nrow` per row)
- `train/student_input`: grid of augmented inputs (same layout)

### Val (TensorBoard, once per epoch in `on_train_epoch_end`)
- **Losses**: `val/loss`, `val/loss_sum`, `val/loss_sp`, `val/cos_sum`, `val/cos_sp`, `val/mse_sum`, `val/mse_sp`, `val/grad`
- **Summary diagnostics**: `val/summary_cos_{mean,pXX}`, `val/retrieval_top1`, `val/retrieval_mrr`, `val/summary_linear_cka`, `val/summary_mse_mean`, `val/summary_norm_ratio`, `val/summary_{mean,std}_abs_diff`
- **Spatial diagnostics**: `val/spatial_cos_{mean,pXX}`, `val/spatial_mse_mean`, `val/spatial_norm_ratio`, `val/spatial_style_gram_mse`, `val/spatial_energy_ratio`, `val/spatial_meanD_corr`
- **Edge/HF diagnostics**: `val/edge_align_corr_{mean,pXX}`, `val/hf_cos_mean`, `val/hf_mse_mean`, `val/hf_energy_ratio`
- **Spatial compare images** (batches matching `logging.val_spatial_compare.batch_mod/max_batches`): cosine map, L2 map, side-by-side channel grid (max resolution `logging.val_spatial_compare.max_side`)
- Quantile suffixes (pXX) are derived from `logging.quantiles` list (e.g. `[0.05, 0.95]` → p05, p95)

### Console
- `[MEM]` prints every `logging.mem_track_interval` steps with allocated/reserved GPU MB
- `[VAL]` once per epoch: `epoch=E step=S loss=X.XXXX loss_sum=X.XXXX loss_sp=X.XXXX (aggregated over N val batches)`

## Checkpointing

- Once per epoch, only if val loss improved over best.
- Saves student, summary head, and spatial head state dicts (not teacher, not optimizer).
- Path: `{experiment.root}/{exp_name}/checkpoints/epoch_E_step_S_val_loss_X.XXX.pth`

## Debug Mode

When `mode=debug`: the `debug:` overlay from config.yaml is merged (by default: `data.train_cap=5000`, `data.val_cap=1000`, `logging.eval_every=5`, `logging.eval_batches=5`), and `_debug` is appended to the experiment suffix. All debug overrides are configurable in the YAML rather than hardcoded.

## Memory Management

- Teacher stored via `object.__setattr__` (not an nn.Module submodule).
- Manual optimization: backward called immediately per micro-batch, graph freed before next step.
- All val diagnostics computed on CPU after moving tensors off GPU.
- `torch.cuda.empty_cache()` after each val epoch.
- Explicit `del` of all GPU intermediates in both train and val steps.
