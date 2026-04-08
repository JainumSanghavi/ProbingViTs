# Probing ViTs for Boundary Detection

Probing Vision Transformer (ViT-Base) hidden representations to detect object boundaries, using the BSDS500 dataset. Part of the Actionable Interpretability course.

## Project Status

**Pipeline: Fully functional end-to-end.**

All steps below have been run successfully. Trained checkpoints, metrics, and figures are already generated in `results/`.

## Project Structure

```
ProbingViTs/
├── configs/default.yaml          # All hyperparameters and paths
├── scripts/
│   ├── run_all.py                # End-to-end orchestrator
│   ├── download_bsds500.py       # Step 1: Download BSDS500
│   ├── preprocess_labels.py      # Step 2: Patch-level boundary labels
│   ├── extract_hidden_states.py  # Step 3: Extract ViT hidden states
│   ├── train_probes.py           # Step 4: Train probes (linear, MLP)
│   ├── evaluate.py               # Step 5: Evaluate on test set
│   └── visualize.py              # Step 6: Generate all figures
├── src/
│   ├── data/                     # Dataset loading, transforms, patch labels
│   ├── models/vit_extractor.py   # ViT hidden state extraction
│   ├── probes/
│   │   ├── linear_probe.py       # LinearProbe, MLPProbe, ConvProbe
│   │   └── baselines.py          # Baseline methods
│   ├── training/trainer.py       # Training loop with early stopping
│   ├── evaluation/metrics.py     # Accuracy, F1, precision, recall, AP
│   ├── visualization/
│   │   ├── qualitative.py        # Prediction overlays + heatmap grid
│   │   ├── layerwise_plots.py    # F1/accuracy across layers
│   │   └── pr_curves.py          # Precision-recall curves
│   └── utils/                    # Config loading, device selection
├── data/                         # Raw BSDS500, processed labels, cached hidden states
└── results/
    ├── checkpoints/              # 52 trained probe checkpoints
    ├── metrics/                  # test_results.json, baseline_results.json, training_results.json
    └── figures/                  # All generated plots (see below)
```

## How to Run

```bash
# Full pipeline (download -> preprocess -> extract -> train -> evaluate -> visualize)
python scripts/run_all.py

# Or run individual steps:
python scripts/download_bsds500.py
python scripts/preprocess_labels.py
python scripts/extract_hidden_states.py
python scripts/train_probes.py
python scripts/evaluate.py
python scripts/visualize.py

# Skip already-completed steps:
python scripts/run_all.py --skip-download --skip-extract --skip-train
```

## Model & Data

- **Model:** `google/vit-base-patch16-224-in21k` (ViT-Base, 768-dim hidden states, 13 layers: embedding + 12 transformer blocks)
- **Dataset:** BSDS500 (200 train / 100 val / 200 test images)
- **Task:** Binary classification per patch token — is this 16x16 patch on an object boundary?
- **Probes:** Linear (769 params) and MLP (197K params), trained on all 13 layers
- **Controls:** Pretrained vs. random-init ViT comparison

## Generated Figures

| File | Description |
|------|-------------|
| `layerwise_f1.png` | F1 score across layers 0-12 for each probe type |
| `layerwise_accuracy.png` | Accuracy across layers 0-12 |
| `pretrained_vs_random.png` | Pretrained vs. random-init ViT comparison |
| `pr_curves.png` | Precision-recall curves |
| `qualitative_layer06.png` | Side-by-side: original, GT, predicted boundaries (layer 6) |
| `heatmap_grid.png` | **Multi-layer heatmap grid** — sigmoid probability overlays across layers 0, 3, 6, 9, 12 for 5 test images |

## Recent Changes

- **Added `visualize_heatmap_grid`** in `src/visualization/qualitative.py`: Creates a grid showing how boundary detection evolves across ViT layers. Each cell overlays the probe's sigmoid probabilities (upsampled from 14x14 to 224x224 via bilinear interpolation) as a `hot` colormap on the original image, with a shared 0-1 probability colorbar.
- **Updated `scripts/visualize.py`** to call `visualize_heatmap_grid` after existing visualizations.

## Possible Next Steps

- Add ConvProbe training/evaluation (architecture exists in `linear_probe.py` but not yet trained)
- Per-layer analysis of what boundary types each layer captures (edges, contours, texture boundaries)
- Attention head analysis
- Fine-grained metrics (boundary recall by edge strength)
