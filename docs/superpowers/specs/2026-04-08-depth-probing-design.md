# Depth Probing Experiment Design

**Date:** 2026-04-08
**Project:** Probing Vision Transformers to Extract Structural and Spatial Information
**Author:** Jainum Sanghavi

---

## Overview

This document specifies the design for a depth probing experiment using NYU Depth V2, extending the existing boundary detection probing work (BSDS500). The experiment asks whether ViT-B/16 intermediate representations linearly encode depth — a genuinely spatial property — and at which layers this encoding emerges. Results will be compared directly against the boundary experiment to assess whether spatial and structural features are encoded at the same or different layers.

---

## Architecture

The experiment mirrors the boundary detection pipeline exactly but is implemented as a **separate, parallel experiment** sharing only the `src/` library. No existing boundary code is modified.

### New Files

```
ProbingViTs/
├── configs/
│   └── depth.yaml                         # Depth-specific hyperparameters
├── scripts/depth/
│   ├── run_depth.py                        # End-to-end orchestrator
│   ├── download_nyu.py                     # HuggingFace Hub download
│   ├── preprocess_depth_labels.py          # Mean depth per patch + normalization
│   ├── extract_hidden_states_depth.py      # Hidden state extraction (reuses ViTExtractor)
│   ├── train_probes_depth.py               # Probe training (reuses LinearProbe/MLPProbe)
│   ├── evaluate_depth.py                   # MAE/RMSE evaluation per layer
│   └── visualize_depth.py                  # All figures
├── src/data/
│   ├── nyu_depth.py                        # HuggingFace dataset loader + preprocessing
│   └── depth_labels.py                     # Mean depth aggregation + normalization
└── results/depth/
    ├── checkpoints/                        # 52 trained probe checkpoints
    ├── metrics/                            # depth_test_results.json, etc.
    └── figures/                            # All generated plots
```

### Reused Unchanged

- `src/models/vit_extractor.py` — ViT hidden state extraction
- `src/probes/linear_probe.py` — LinearProbe, MLPProbe
- `src/training/trainer.py` — Training loop with early stopping
- `src/utils/` — Config loading, device selection

---

## Data

**Dataset:** NYU Depth V2, loaded via HuggingFace Hub (`sayakpaul/nyu_depth_v2`). No manual download required — the `datasets` library handles acquisition automatically.

**Split:** 795 train / 654 test (as defined by the dataset). A validation set of ~120 images (~15% of train) is carved out for early stopping.

**Preprocessing:**
- RGB images resized to 224×224 and preprocessed with the ViT image processor (same as boundary experiment)
- Depth maps resized to 224×224 using **bilinear interpolation** (appropriate for continuous values, unlike the nearest-neighbor used for binary boundary maps)

**Patch labels:**
- Each 16×16 patch → mean depth of all pixels in that patch → flat `(196,)` float array
- Normalized by dividing by 10.0 (NYU Depth V2 max depth ~10 meters) → values in `[0, 1]`
- MAE reported in normalized units; multiply by 10 to recover meters

**Caching:** Hidden states saved to `data/depth_cached/` as `.pt` files, one per image per model initialization (pretrained/random), matching the format of `data/cached/`.

---

## Probes & Training

### Architecture

| Probe | Architecture | Output |
|-------|-------------|--------|
| LinearProbe | 768 → 1 | Linear (no activation) |
| MLPProbe | 768 → 256 → ReLU → 1 | Linear (no activation) |

The sigmoid activation used in the boundary experiment is removed — raw linear output is appropriate for regression.

### Training

- **Loss:** `MSELoss`
- **Optimizer:** Adam, lr=0.001, weight decay=1e-4
- **Batch size:** 512
- **Max epochs:** 100
- **Early stopping:** patience=10, monitored on `val_mae` (minimize)
- **Seed:** 42

### Controls

Both pretrained and randomly initialized ViT-B/16 (`google/vit-base-patch16-224-in21k`) are probed, producing the same 52 checkpoints as the boundary experiment (13 layers × 2 probe types × 2 model inits).

---

## Evaluation

**Primary metric:** MAE (Mean Absolute Error) — `mean(|predicted - actual|)` over all patches in the test set, reported per layer per probe type.

**Secondary metric:** RMSE — reported alongside MAE.

Results saved to `results/depth/metrics/depth_test_results.json` with the same structure as `results/metrics/test_results.json`.

---

## Figures

| File | Description |
|------|-------------|
| `layerwise_mae.png` | MAE across layers 0–12 for linear/MLP probes, pretrained vs random |
| `pretrained_vs_random_depth.png` | Pretrained vs random ViT comparison for depth probing |
| `qualitative_depth_layer06.png` | Side-by-side: original image, GT depth map, predicted depth map (layer 6) |
| `depth_heatmap_grid.png` | Multi-layer predicted depth overlays (layers 0, 3, 6, 9, 12) using `plasma` colormap |
| `cross_task_layerwise.png` | Normalized MAE (depth) and F1 (boundary) overlaid on same axis — key paper figure |
| `depth_scatter_best_layer.png` | Predicted vs actual depth scatter at the layer with lowest test MAE (linear probe, pretrained ViT) |
| `mae_by_depth_bin.png` | MAE broken down by near (0–0.3), mid (0.3–0.6), far (0.6–1.0) normalized depth ranges |
| `patch_error_map.png` | Spatial heatmap of per-patch MAE for a selection of test images |

**Colormap:** `plasma` for all depth visualizations (perceptually uniform, standard for depth).

---

## Key Differences from Boundary Experiment

| Aspect | Boundary (BSDS500) | Depth (NYU Depth V2) |
|--------|-------------------|----------------------|
| Task | Binary classification | Regression |
| Label | Binary per patch | Mean depth per patch, normalized [0,1] |
| Loss | BCEWithLogitsLoss | MSELoss |
| Probe output | Sigmoid | Linear |
| Primary metric | F1, accuracy | MAE |
| Depth resize | Nearest neighbor | Bilinear |
| Class imbalance | pos_weight in loss | N/A |

---

## Paper Relevance

The central claim of the paper is that ViTs encode both structural (boundary) and spatial (depth) properties without explicit supervision, and that these properties may emerge at different layers. The `cross_task_layerwise.png` figure is the primary visual for this claim. The pretrained vs random control establishes that encoding is learned, not random. The patch error map and depth bin analysis provide additional interpretability evidence for the methods/results section.
