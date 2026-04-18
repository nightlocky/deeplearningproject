# OCT Anomaly Detection Project

This project explores anomaly detection on OCT retinal images using several deep learning approaches, including reconstruction-based autoencoders, feature-space autoencoders, centroid-distance ViT features, and U-Net-based anomaly scoring pipelines.

## Project Goal

The main goal is to distinguish normal OCT scans from pathological scans by learning patterns from normal data and then identifying abnormal deviations at test time. The repository includes both baseline models and a tuning pipeline for selecting the strongest final model.

## Final Report

The final project report can be accessed here: [Deep Learning Report](./final_report.pdf)

## Dataset

The project uses OCT images with one normal class and multiple anomaly classes.

The experiments are based on the public OCT2017 retinal dataset, but the repository uses a modified version of the original split. In particular, the last `9750` normal images from the original training split were moved into the test-side normal pool so that the final evaluation uses a larger held-out set of normal scans.

| Split      | Normal | DRUSEN | DME | CNV |
| ---------- | -----: | -----: | --: | --: |
| Train      | 40,000 |      0 |   0 |   0 |
| Test       |  5,000 |     75 |  75 |  75 |
| Validation |  5,000 |     75 |  75 |  75 |

The dataset is not included in this repository due to storage restrictions. Before running the code, place the OCT dataset under the paths expected by [src/config.py](C:/Users/anged/Desktop/deeplearningproject/src/config.py):

```text
/workspace/data/train
/workspace/data/test
```

The training split should contain only normal images for the anomaly-detection training pipeline, while the test split should contain normal and anomalous scans from `CNV`, `DME`, and `DRUSEN`.

### Recreating the Dataset Split Used in This Project

1. Download the original OCT2017 dataset from its public source.
2. Keep the anomaly classes `CNV`, `DME`, and `DRUSEN` in the test-side pool.
3. In the original `train/NORMAL` folder, move the last `9750` normal images into the test-side normal pool.
4. Keep the remaining normal images in the training folder. The code will then sample `40000` training-normal images from this remaining pool during training.
5. Place the resulting folders under:

```text
/workspace/data/train
/workspace/data/test
```

The code then samples the working split used in this project as:

- training: `40000` normal images
- validation: `5000` normal + `75` each of `CNV`, `DME`, and `DRUSEN`
- test: `5000` normal + `75` each of `CNV`, `DME`, and `DRUSEN`

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
- MLP hidden layers: `[512, 256, 128, 64]`
- MLP dropout: `0.1`

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

## Training the Final Model From Scratch

To retrain the final best model from scratch, first review the tuning settings in [config_tune.py](C:/Users/anged/Desktop/deeplearningproject/src/models/best_model_tuning/config_tune.py). The main fields are:

- `CURRENT_PHASE`: selects which stage of the tuning pipeline to run
- `BACKBONES`: candidate encoder backbones evaluated in Phase 1
- `LOSS_ALPHAS`: candidate reconstruction-loss weightings evaluated in Phase 2
- `MLP_ARCHITECTURES`: candidate latent MLP hidden-layer settings evaluated in Phase 3
- `MLP_DROPOUTS`: candidate dropout values evaluated in Phase 3
- `BEST_BACKBONE_SO_FAR`: best backbone carried from Phase 1 into Phase 2
- `BEST_AE_WEIGHTS_PATH`: best autoencoder checkpoint carried from Phase 2 into Phase 3
- `RANDOM_SEED`: reproducibility setting for the tuning script

The final best-performing configuration from this tuning process was:

- encoder backbone: `resnet50`
- reconstruction loss weighting: `0.5`
- encoder output latent dimension: `2048`
- MLP hidden layers: `[512, 256, 128, 64]`
- MLP dropout: `0.1`
- random seed: `42`

These values describe the final saved reproducible model currently stored in [src/inference/best_model](C:/Users/anged/Desktop/deeplearningproject/src/inference/best_model). They should be distinguished from the wider Phase 3 search space, where the tuning script evaluates multiple candidate MLP architectures and dropout settings:

- `MLP_ARCHITECTURES = [[1024, 512, 256], [1024, 256], [512, 256, 128, 64], [256, 64]]`
- `MLP_DROPOUTS = [0.1, 0.3, 0.5]`

The training workflow is phase-based:

1. Set `CURRENT_PHASE = 1` to test candidate U-Net backbones.
2. Set `CURRENT_PHASE = 2` after updating `BEST_BACKBONE_SO_FAR` to test reconstruction-loss weightings.
3. Set `CURRENT_PHASE = 3` after updating `BEST_AE_WEIGHTS_PATH` to test MLP architectures and dropout values.

Run the same command for each phase:

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

Keep these files inside [src/inference/best_model](C:/Users/anged/Desktop/deeplearningproject/src/inference/best_model):

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

This loader reproduces the saved architecture exactly by rebuilding:

- the `resnet50` U-Net autoencoder
- the latent-space MLP with hidden layers `[512, 256, 128, 64]`
- the saved anomaly threshold from [best_model_metadata.json](C:/Users/anged/Desktop/deeplearningproject/src/inference/best_model/best_model_metadata.json)

To reproduce the reported final performance results shown in the PDF report, load the saved weights above and evaluate them on the same dataset split described in [src/config.py](C:/Users/anged/Desktop/deeplearningproject/src/config.py) and [src/dataLoader/data_loader.py](C:/Users/anged/Desktop/deeplearningproject/src/dataLoader/data_loader.py). The reported figures in the repository were generated from the same best-model tuning pipeline and include:

- `test_confusion_matrix.png`
- `test_anomaly_score_distribution.png`
- `prediction_breakdown_by_class.png`
- false negative sample visualizations under `samples/`

If the large `.pth` files cannot be hosted directly on GitHub, store them externally and place them back into [src/inference/best_model](C:/Users/anged/Desktop/deeplearningproject/src/inference/best_model) before running the commands above.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```
