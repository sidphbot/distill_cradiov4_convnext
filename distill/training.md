# Distillation Training Workflow

## Architecture

- **Teacher**: C-RADIOv4-H (ViT-Huge), frozen, fp16. Produces summary embedding `(B, Ct)` and spatial tokens `(B, T, Dt)`.
- **Student**: ConvNeXt-small (or tiny), `features_only` with stages 2+3. Stage 3 pooled → `SummaryHead → (B, Ct)`. Stage 2 → `SpatialHead → (B, Dt, Ht, Wt)`.
- Both operate at 416×416 input, patch_size=16 → Ht=Wt=26, T=676.

## Data Pipeline

1. `ImagePathDataset` returns raw file paths; `OnlineCollate` loads + resizes at batch time.
2. **Resize**: 50/50 deterministic (hash of `seed:path`) choice between aspect-ratio-preserving pad (black canvas) and squash to 416×416. Seed changes each epoch for variation.
3. Collate returns `(img_u8, pil_rs, keys)` — uint8 tensor for student, PIL list for teacher processor, paths for debugging.
4. Train/val split is deterministic via stable hash on path.

## Training Step (Manual Optimization)

Each call to `training_step` is one micro-batch:

1. **Teacher forward** (`@torch.no_grad`, fp16 autocast): teacher processor → teacher model → `(t_summary, t_spatial_tokens)`.
2. **Ramp loss weights** (see Ramps below).
3. **Student augmentation** on CPU (see Augmentations below), then ImageNet normalize in-place, single `.to(device)` copy.
4. **Student forward + loss** under fp16 autocast.
5. **Extract log scalars** as Python floats before backward.
6. **`scaler.scale(loss / grad_accum).backward()`** — graph freed immediately.
7. **Delete all GPU intermediates** (`del loss, out, s_sum, ...`).
8. Every `grad_accum` (4) micro-steps: unscale → clip_grad_norm(1.0) → scaler.step → scaler.update → zero_grad(set_to_none=True).

## Loss Function

```
loss = λ_summary * loss_sum + λ_spatial * loss_sp
loss_sum = cosine_loss(s_sum, t_sum) + λ_mse * mse(s_sum, t_sum)
loss_sp  = cosine_loss(s_tokens, t_tokens) + λ_sp_mse * mse(s_sp, t_sp) + λ_grad * sobel_grad_loss(s_sp, t_sp)
```

- `cosine_loss`: `(1 - cos_sim).mean()` on L2-normalized vectors.
- `sobel_grad_loss`: Sobel edge maps on both student/teacher spatial feature maps, Charbonnier-robust L1 on the difference.
- Spatial norm capping at 50.0 on student output to prevent hotspots.

## Ramps

All use `ramp_linear(step, warmup_steps, start, end)` — linear interpolation clamped at end.

| Weight | Start | End | Warmup | Purpose |
|---|---|---|---|---|
| `λ_sp_mse` | 0.5 | 1.5 | 20% of epoch | Ease spatial MSE pressure early |
| `λ_grad` | 0.1 | 1.3 | 35% of epoch | Let coarse structure form before enforcing edges |
| `aug_p` | 0.0 | 0.6 | 4000 steps | Let student learn clean mapping first, then add robustness |

## Augmentations (Student Only)

Applied on CPU to uint8→float32 before normalization. Teacher always sees clean resized images.

Per-sample gated by `aug_p` (ramped 0→0.6 over 4000 steps). If gated on, each aug applied independently:

| Augmentation | Probability | Range |
|---|---|---|
| Brightness | 0.4 | ±0.1 multiplicative |
| Contrast | 0.4 | ±0.1 around mean |
| Color jitter | 0.3 | hue ±0.02, saturation ±0.1 |
| Gaussian blur | 0.3 | kernel 3 or 5, σ 0.1–0.5 |
| ISO noise | 0.3 | additive Gaussian, σ=0.02 |

## Validation

- Runs every `eval_every * grad_accum` training batches (e.g. debug: every 20 batches = every 5 optimizer steps).
- `limit_val_batches` caps how many val batches per run (debug: 5, full: 50).
- `validation_step` is wrapped in `@torch.no_grad()`. Forward + loss on GPU, then everything moved to CPU for diagnostics. GPU freed + `empty_cache()` after each val epoch.
- **Accumulators persist across all val runs within a training epoch** — results are only reported once in `on_train_epoch_end`.

## Logging

### Train (TensorBoard, every `log_every` steps via Lightning)
- `train/loss`, `train/loss_sum`, `train/loss_sp`, `train/loss_grad`
- `train/cos_sum`, `train/cos_sp`, `train/mse_sum`, `train/mse_sp`
- `train/mse_sp_w`, `train/grad_w`, `train/aug_p` (current ramp values)

### Train images (every `log_every * 5` steps)
- `train/teacher_input`: 8-image grid of un-augmented inputs
- `train/student_input`: 8-image grid of augmented inputs

### Val (TensorBoard, once per epoch in `on_train_epoch_end`)
- **Losses**: `val/loss`, `val/loss_sum`, `val/loss_sp`, `val/loss_grad`, `val/cos_sum`, `val/cos_sp`, `val/mse_sum`, `val/mse_sp`
- **Summary diagnostics**: `val/summary_cos_{mean,p05,p95}`, `val/retrieval_top1`, `val/retrieval_mrr`, `val/summary_linear_cka`, `val/summary_mse_mean`, `val/summary_norm_ratio`, `val/summary_{mean,std}_abs_diff`
- **Spatial diagnostics**: `val/spatial_cos_{mean,p05,p95}`, `val/spatial_mse_mean`, `val/spatial_norm_ratio`, `val/spatial_style_gram_mse`, `val/spatial_energy_ratio`, `val/spatial_meanD_corr`
- **Edge/HF diagnostics**: `val/edge_align_corr_{mean,p05,p95}`, `val/hf_cos_mean`, `val/hf_mse_mean`, `val/hf_energy_ratio`
- **Spatial compare images** (first few val batches): cosine map, L2 map, side-by-side channel grid

### Console (once per epoch)
- `[VAL] epoch=E step=S loss=X.XXXX loss_sum=X.XXXX loss_sp=X.XXXX (aggregated over N val batches)`

## Checkpointing

- Once per epoch, only if val loss improved over best.
- Saves student, summary head, and spatial head state dicts (not teacher, not optimizer).
- Path: `{exp_root}/{exp_name}/checkpoints/epoch_E_step_S_val_loss_X.XXX.pth`

## Debug Mode

When `--mode debug`: `train_cap=5000`, `val_cap=1000`, `eval_every=5`, `eval_batches=5`, suffix `_debug` appended to experiment name. Config overrides happen in `parse_args`, not in data module.

## Memory Management

- Teacher stored via `object.__setattr__` (not an nn.Module submodule).
- Manual optimization: backward called immediately per micro-batch, graph freed before next step.
- All val diagnostics computed on CPU after moving tensors off GPU.
- `torch.cuda.empty_cache()` after each val epoch.
- Explicit `del` of all GPU intermediates in both train and val steps.
