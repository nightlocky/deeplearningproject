# OCT Anomaly Detection Project

This project explores anomaly detection on OCT retinal images using several deep learning approaches, including reconstruction-based autoencoders, feature-space autoencoders, centroid-distance ViT features, and U-Net-based anomaly scoring pipelines.

## Project Goal

The main goal is to distinguish normal OCT scans from pathological scans by learning patterns from normal data and then identifying abnormal deviations at test time. The repository includes both baseline models and a tuning pipeline for selecting the strongest final model.

## Dataset

The project uses OCT images with one normal class and multiple anomaly classes.

| Split | Normal | DRUSEN |    DME |    CNV |
| ----- | -----: | -----: | -----: | -----: |
| Train | 51,140 |  8,616 | 11,348 | 37,205 |
| Test  |    250 |    250 |    250 |    250 |

### Preprocessing

All models use the shared preprocessing defined in [src/dataLoader/data_loader.py](C:/Users/anged/Desktop/deeplearningproject/src/dataLoader/data_loader.py):

- resize to `224 x 224`
- convert to grayscale
- convert to tensor with `ToTensor()`

The shared runtime settings live in [src/config.py](C:/Users/anged/Desktop/deeplearningproject/src/config.py).

## Repository Structure

- [src/config.py](C:/Users/anged/Desktop/deeplearningproject/src/config.py): central paths, dataset sizes, hardware, and image settings
- [src/dataLoader](C:/Users/anged/Desktop/deeplearningproject/src/dataLoader): dataset loading and class-balanced sampling
- [src/helper](C:/Users/anged/Desktop/deeplearningproject/src/helper): MLflow, early stopping, and visualization utilities
- [src/models](C:/Users/anged/Desktop/deeplearningproject/src/models): standalone training and evaluation scripts
- [src/models/best_model_tuning](C:/Users/anged/Desktop/deeplearningproject/src/models/best_model_tuning): phased tuning pipeline for the final best model
- [src/inference/best_model](C:/Users/anged/Desktop/deeplearningproject/src/inference/best_model): reproducible loading files for the saved best model
- [graphs](C:/Users/anged/Desktop/deeplearningproject/graphs): generated figures and analysis plots
- [mlruns](C:/Users/anged/Desktop/deeplearningproject/mlruns): MLflow tracking outputs

## Models Implemented

- [pure_autoencoder.py](C:/Users/anged/Desktop/deeplearningproject/src/models/pure_autoencoder.py): convolutional autoencoder trained directly on OCT image reconstruction
- [pure_vit.py](C:/Users/anged/Desktop/deeplearningproject/src/models/pure_vit.py): anomaly detection using centroid distance over pretrained ViT features
- [vit_autoencoder.py](C:/Users/anged/Desktop/deeplearningproject/src/models/vit_autoencoder.py): autoencoder trained on ViT feature embeddings
- [patchcore_autoencoder.py](C:/Users/anged/Desktop/deeplearningproject/src/models/patchcore_autoencoder.py): PatchCore-style ResNet features reconstructed with a convolutional feature autoencoder
- [unet_cnn.py](C:/Users/anged/Desktop/deeplearningproject/src/models/unet_cnn.py): U-Net autoencoder followed by a CNN over spatial residual maps
- [unet_mlp.py](C:/Users/anged/Desktop/deeplearningproject/src/models/unet_mlp.py): U-Net autoencoder followed by an MLP over pooled latent features

## Best Model

The current best tuned configuration comes from the phased tuning pipeline:

- autoencoder backbone: `resnet50`
- tuning loss alpha: `0.5`
- latent dimension: `2048`
- MLP hidden layers: `[256, 64]`
- MLP dropout: `0.3`
- decision threshold: `0.3208799362182617`

Saved best-model artifacts and metadata are stored in [src/inference/best_model](C:/Users/anged/Desktop/deeplearningproject/src/inference/best_model):

- [best_model_metadata.json](C:/Users/anged/Desktop/deeplearningproject/src/inference/best_model/best_model_metadata.json)
- [load_best_model.py](C:/Users/anged/Desktop/deeplearningproject/src/inference/best_model/load_best_model.py)
- `best_final_mlp.pth`
- `loss_functions_0.5_best_ae.pth`

## Training and Tuning

### Run a standalone model

Examples:

```bash
python -m src.models.pure_autoencoder
python -m src.models.pure_vit
python -m src.models.vit_autoencoder
python -m src.models.patchcore_autoencoder
python -m src.models.unet_cnn
python -m src.models.unet_mlp
```

### Run the tuning pipeline

The tuning workflow is managed by [best_model_script.py](C:/Users/anged/Desktop/deeplearningproject/src/models/best_model_tuning/best_model_script.py) and [config_tune.py](C:/Users/anged/Desktop/deeplearningproject/src/models/best_model_tuning/config_tune.py).

Phases:

1. test candidate U-Net backbones
2. test different autoencoder loss weights
3. test multiple MLP architectures and dropout settings

Run:

```bash
python -m src.models.best_model_tuning.best_model_script
```

## Evaluation and Visualizations

The project reports the following evaluation metrics across the anomaly detection pipelines:

- precision
- recall
- F1-score
- specificity
- AUC-ROC
- AUC-PR
- confusion matrix
- threshold-based anomaly score distributions

Generated visual outputs include:

- loss curves
- confusion matrices
- anomaly score distributions
- standardized sample visualizations:
  - feature-space models: original image, overlay, error heatmap
  - reconstruction models: original image, reconstructed image, error heatmap

These figures are created through [visualization_helper.py](C:/Users/anged/Desktop/deeplearningproject/src/helper/visualization_helper.py).

## Reproducible Loading Without Retraining

The best tuned model can be rebuilt directly from saved weights and metadata.

### What to keep together

Keep these files together inside [src/inference/best_model](C:/Users/anged/Desktop/deeplearningproject/src/inference/best_model):

- `best_final_mlp.pth`
- `loss_functions_0.5_best_ae.pth`
- `best_model_metadata.json`

Check that the saved model files load correctly:

```bash
python -m src.inference.best_model.load_best_model --metadata src/inference/best_model/best_model_metadata.json --check-only
```

Predict one image directly:

```bash
python -m src.inference.best_model.load_best_model --metadata src/inference/best_model/best_model_metadata.json --image path/to/image.jpeg
```

This loader rebuilds:

- the `segmentation_models_pytorch.Unet` autoencoder
- the latent feature extraction path using `AdaptiveAvgPool2d((1, 1))`
- the tuned `LatentMLP`
- the saved threshold-based anomaly decision rule

This means the loader does not retrain the model. It rebuilds the saved architecture, loads the saved weights, applies the stored threshold, and can return a simple `Normal` or `Anomaly` prediction for a new image.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended environment:

- Python 3.11
- CUDA-enabled PyTorch environment for training speed
