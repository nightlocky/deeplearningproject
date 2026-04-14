# Best Model Loading

This folder is for reproducible loading of the tuned best model without retraining.

## What to save

Keep these three files together:

1. `best_final_mlp.pth`
2. The matching best autoencoder weights `.pth`
3. A metadata JSON file based on `metadata.template.json`

## Recommended workflow

1. Copy `best_final_mlp.pth` from RunPod into a persistent location in this project.
2. Use `best_model_metadata.json` as the finalized metadata file.
3. Update it if any artifact paths or winning-run details change.
4. Load the model with:

```bash
python -m src.inference.best_model.load_best_model --metadata src/inference/best_model/best_model_metadata.json --check-only
```

## What the loader rebuilds

- A `segmentation_models_pytorch.Unet` autoencoder
- The latent pooling step with `AdaptiveAvgPool2d((1, 1))`
- The tuned `LatentMLP`
- The saved threshold for anomaly decisions

## Reproducibility note

`best_final_mlp.pth` by itself is not enough. You also need the matching autoencoder weights and the metadata settings that define the backbone, latent dimension, hidden-layer layout, dropout, preprocessing, and threshold.
