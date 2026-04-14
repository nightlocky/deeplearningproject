import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src import config
from src.models.best_model_tuning.latent_mlp import LatentMLP


@dataclass
class BestModelBundle:
    autoencoder: nn.Module
    mlp: nn.Module
    threshold: float
    metadata: dict[str, Any]
    device: torch.device


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(config.PROJECT_ROOT) / path


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_autoencoder(metadata: dict[str, Any]) -> nn.Module:
    arch = metadata["architecture"]
    return smp.Unet(
        encoder_name=arch["backbone"],
        encoder_weights=arch.get("encoder_weights"),
        in_channels=arch.get("in_channels", 1),
        classes=arch.get("classes", 1),
        activation=arch.get("activation", "sigmoid"),
    )


def infer_latent_dim(autoencoder: nn.Module, image_size: int, device: torch.device) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, 1, image_size, image_size, device=device)
        features = autoencoder.encoder(dummy)
        pooled = nn.AdaptiveAvgPool2d((1, 1))(features[-1]).flatten(1)
    return int(pooled.shape[1])


def load_best_model(metadata_path: str | Path, device: torch.device | None = None) -> BestModelBundle:
    metadata = load_metadata(metadata_path)
    device = device or config.DEVICE

    ae_weights_path = _resolve_path(metadata["artifacts"]["ae_weights_path"])
    mlp_weights_path = _resolve_path(metadata["artifacts"]["mlp_weights_path"])

    if not ae_weights_path.exists():
        raise FileNotFoundError(f"Autoencoder weights not found: {ae_weights_path}")
    if not mlp_weights_path.exists():
        raise FileNotFoundError(f"MLP weights not found: {mlp_weights_path}")

    autoencoder = build_autoencoder(metadata).to(device)
    autoencoder.load_state_dict(torch.load(ae_weights_path, map_location=device, weights_only=True))
    autoencoder.eval()

    image_size = int(metadata["preprocessing"]["image_size"])
    latent_dim = metadata["architecture"].get("latent_dim") or infer_latent_dim(autoencoder, image_size, device)
    hidden_layers = metadata["architecture"]["mlp_hidden_layers"]
    dropout = float(metadata["architecture"]["mlp_dropout"])

    mlp = LatentMLP(input_dim=latent_dim, hidden_layers=hidden_layers, dropout_rate=dropout).to(device)
    mlp.load_state_dict(torch.load(mlp_weights_path, map_location=device, weights_only=True))
    mlp.eval()

    threshold = metadata["inference"]["threshold"]
    if isinstance(threshold, str):
        raise ValueError("metadata inference.threshold must be filled with a numeric value before loading.")

    return BestModelBundle(
        autoencoder=autoencoder,
        mlp=mlp,
        threshold=float(threshold),
        metadata=metadata,
        device=device,
    )


def extract_latent_features(bundle: BestModelBundle, images: torch.Tensor) -> torch.Tensor:
    pool = nn.AdaptiveAvgPool2d((1, 1))
    images = images.to(bundle.device)
    with torch.no_grad():
        encoder_features = bundle.autoencoder.encoder(images)
        return pool(encoder_features[-1]).flatten(1)


def predict_scores(bundle: BestModelBundle, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latents = extract_latent_features(bundle, images)
        return bundle.mlp(latents)


def predict_labels(bundle: BestModelBundle, images: torch.Tensor) -> torch.Tensor:
    scores = predict_scores(bundle, images)
    return (scores > bundle.threshold).int()


def main():
    parser = argparse.ArgumentParser(description="Load the tuned best model without retraining.")
    parser.add_argument("--metadata", required=True, help="Path to the best model metadata JSON file.")
    parser.add_argument("--check-only", action="store_true", help="Only verify that the saved artifacts can be rebuilt and loaded.")
    args = parser.parse_args()

    bundle = load_best_model(args.metadata)
    print(f"Loaded autoencoder backbone: {bundle.metadata['architecture']['backbone']}")
    print(f"Loaded MLP hidden layers: {bundle.metadata['architecture']['mlp_hidden_layers']}")
    print(f"Loaded threshold: {bundle.threshold}")

    if args.check_only:
        print("Model artifacts loaded successfully.")


if __name__ == "__main__":
    main()
