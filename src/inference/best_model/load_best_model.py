import argparse
import json
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src import config
from src.models.best_model_tuning.latent_mlp import LatentMLP


def _read_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _project_path(path_str):
    path = Path(path_str)
    return path if path.is_absolute() else Path(config.PROJECT_ROOT) / path


def _build_autoencoder(arch):
    return smp.Unet(
        encoder_name=arch["backbone"],
        encoder_weights=arch.get("encoder_weights"),
        in_channels=arch.get("in_channels", 1),
        classes=arch.get("classes", 1),
        activation=arch.get("activation", "sigmoid"),
    )


def _get_latent_dim(autoencoder, image_size, device):
    with torch.no_grad():
        dummy = torch.zeros(1, 1, image_size, image_size, device=device)
        deepest_feature = autoencoder.encoder(dummy)[-1]
        return int(nn.AdaptiveAvgPool2d((1, 1))(deepest_feature).flatten(1).shape[1])


def load_best_model(metadata_path, device=None):
    metadata = _read_metadata(metadata_path)
    device = device or config.DEVICE

    arch = metadata["architecture"]
    artifacts = metadata["artifacts"]
    threshold = metadata["inference"]["threshold"]

    ae_weights_path = _project_path(artifacts["ae_weights_path"])
    mlp_weights_path = _project_path(artifacts["mlp_weights_path"])

    if not ae_weights_path.exists():
        raise FileNotFoundError(f"Autoencoder weights not found: {ae_weights_path}")
    if not mlp_weights_path.exists():
        raise FileNotFoundError(f"MLP weights not found: {mlp_weights_path}")
    if isinstance(threshold, str):
        raise ValueError("metadata inference.threshold must be a numeric value.")

    autoencoder = _build_autoencoder(arch).to(device)
    autoencoder.load_state_dict(torch.load(ae_weights_path, map_location=device, weights_only=True))
    autoencoder.eval()

    image_size = int(metadata["preprocessing"]["image_size"])
    latent_dim = arch.get("latent_dim") or _get_latent_dim(autoencoder, image_size, device)

    mlp = LatentMLP(
        input_dim=latent_dim,
        hidden_layers=arch["mlp_hidden_layers"],
        dropout_rate=float(arch["mlp_dropout"]),
    ).to(device)
    mlp.load_state_dict(torch.load(mlp_weights_path, map_location=device, weights_only=True))
    mlp.eval()

    return {
        "autoencoder": autoencoder,
        "mlp": mlp,
        "threshold": float(threshold),
        "metadata": metadata,
        "device": device,
    }


def extract_latent_features(bundle, images):
    images = images.to(bundle["device"])
    with torch.no_grad():
        deepest_feature = bundle["autoencoder"].encoder(images)[-1]
        return nn.AdaptiveAvgPool2d((1, 1))(deepest_feature).flatten(1)


def predict_scores(bundle, images):
    with torch.no_grad():
        latents = extract_latent_features(bundle, images)
        return bundle["mlp"](latents)


def predict_labels(bundle, images):
    return (predict_scores(bundle, images) > bundle["threshold"]).int()


def main():
    parser = argparse.ArgumentParser(description="Load the tuned best model without retraining.")
    parser.add_argument("--metadata", required=True, help="Path to the best model metadata JSON file.")
    parser.add_argument("--check-only", action="store_true", help="Only verify that the saved artifacts can be loaded.")
    args = parser.parse_args()

    bundle = load_best_model(args.metadata)
    metadata = bundle["metadata"]

    print(f"Loaded autoencoder backbone: {metadata['architecture']['backbone']}")
    print(f"Loaded MLP hidden layers: {metadata['architecture']['mlp_hidden_layers']}")
    print(f"Loaded threshold: {bundle['threshold']}")

    if args.check_only:
        print("Model artifacts loaded successfully.")


if __name__ == "__main__":
    main()
