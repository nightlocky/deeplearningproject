"""
Module: visualization_helper.py
Description: Standardized plotting utilities for OCT anomaly detection.
             Supports both feature-space overlay visualizations and
             image reconstruction visualizations with consistent layouts.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
import torch.nn.functional as F

from src import config

FEATURE_OVERLAY_MODE = "overlay"
RECONSTRUCTION_MODE = "reconstruction"


def _prepare_save_path(save_path: str, model_name: str = None) -> str:
    """
    Builds a save path inside the graphs directory unless an absolute path is given.
    Also creates any missing parent directories for nested paths such as samples/*.png.
    """
    if not save_path:
        return None

    if os.path.isabs(save_path):
        final_path = save_path
    else:
        target_dir = os.path.join(config.GRAPHS_DIR, model_name) if model_name else config.GRAPHS_DIR
        final_path = os.path.join(target_dir, save_path)

    final_dir = os.path.dirname(final_path)
    if final_dir:
        os.makedirs(final_dir, exist_ok=True)
    return final_path


def resolve_visualization_mode(model_name: str = None, view_mode: str = None) -> str:
    """
    Resolves the visualization layout to use.

    Feature-space models such as ViT and PatchCore should use overlays, while
    models that reconstruct the image itself should use the reconstruction mode.
    """
    if view_mode in {FEATURE_OVERLAY_MODE, RECONSTRUCTION_MODE}:
        return view_mode

    name = (model_name or "").lower()
    if any(keyword in name for keyword in ("vit", "patchcore")):
        return FEATURE_OVERLAY_MODE
    return RECONSTRUCTION_MODE


def _to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _normalize_array(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array.astype(np.float32), copy=False)
    min_val = float(np.min(array))
    max_val = float(np.max(array))
    if max_val > min_val:
        return (array - min_val) / (max_val - min_val)
    return np.zeros_like(array, dtype=np.float32)


def _prepare_display_image(sample) -> np.ndarray:
    """
    Converts image-like tensors/arrays into either HxW grayscale or HxWxC format.
    """
    image = np.squeeze(_to_numpy(sample))

    if image.ndim == 3:
        if image.shape[0] in (1, 3):
            image = np.moveaxis(image, 0, -1)
        elif image.shape[-1] not in (1, 3):
            image = np.mean(image, axis=0)

    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    return _normalize_array(image)


def _prepare_grayscale_image(sample) -> np.ndarray:
    image = _prepare_display_image(sample)
    if image.ndim == 3:
        return np.mean(image, axis=-1)
    return image


def _prepare_heatmap(sample) -> np.ndarray:
    """
    Converts heatmap-like tensors/arrays into a normalized 2D map.
    """
    heatmap = np.squeeze(_to_numpy(sample)).astype(np.float32)

    if heatmap.ndim == 0:
        heatmap = np.array([[float(heatmap)]], dtype=np.float32)

    while heatmap.ndim > 2:
        if heatmap.shape[0] in (1, 3):
            heatmap = heatmap[0] if heatmap.shape[0] == 1 else np.mean(heatmap, axis=0)
        elif heatmap.shape[-1] in (1, 3):
            heatmap = heatmap[..., 0] if heatmap.shape[-1] == 1 else np.mean(heatmap, axis=-1)
        else:
            heatmap = np.mean(heatmap, axis=0)

    if heatmap.ndim == 1:
        side = int(np.sqrt(heatmap.shape[0]))
        heatmap = heatmap.reshape(side, side) if side * side == heatmap.shape[0] else heatmap.reshape(1, -1)

    return _normalize_array(heatmap)


def _extract_score(errors, idx: int) -> float:
    scores = _to_numpy(errors)
    if scores.ndim == 0:
        return float(scores)
    return float(scores[idx])


def _show_image(ax, image: np.ndarray, title: str):
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")


def _render_comparison_row(axes_row, raw_image, model_output, score: float, view_mode: str, error_map=None):
    original_image = _prepare_display_image(raw_image)
    _show_image(axes_row[0], original_image, "Original OCT")

    if view_mode == FEATURE_OVERLAY_MODE:
        overlay_map = _prepare_heatmap(model_output)
        _show_image(axes_row[1], original_image, f"Overlay\nScore: {score:.4f}")
        axes_row[1].imshow(overlay_map, cmap="jet", alpha=0.45)
        final_heatmap = _prepare_heatmap(error_map) if error_map is not None else overlay_map
    else:
        reconstruction = _prepare_display_image(model_output)
        _show_image(axes_row[1], reconstruction, f"Reconstructed OCT\nScore: {score:.4f}")
        if error_map is None:
            final_heatmap = np.abs(_prepare_grayscale_image(raw_image) - _prepare_grayscale_image(model_output))
        else:
            final_heatmap = _prepare_heatmap(error_map)

    heatmap_plot = axes_row[2].imshow(_prepare_heatmap(final_heatmap), cmap="jet")
    axes_row[2].set_title("Error Heatmap")
    axes_row[2].axis("off")
    axes_row[2].figure.colorbar(heatmap_plot, ax=axes_row[2], fraction=0.046, pad=0.04)


def plot_loss(losses: list, save_path: str = None, model_name: str = None) -> str:
    """Plots the training loss over epochs in both linear and log scales."""
    epochs = range(1, len(losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, losses, "b-o", label="Training Loss")
    ax1.set_title("Training Loss (Linear Scale)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, losses, "r-o", label="Training Loss (Log)")
    ax2.set_yscale("log")
    ax2.set_title("Training Loss (Logarithmic Scale)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss (Log)")
    ax2.grid(True, which="both", ls="-")
    ax2.legend()

    plt.tight_layout()
    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
    plt.close(fig)
    return final_path


def plot_error_distribution(train_errors, test_normal_errors, test_anomaly_errors, threshold: float, save_path: str = None, model_name: str = None):
    """Plots the distribution of reconstruction errors for detection analysis."""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.histplot(train_errors, bins=50, color="green", label="Train (Normal)", stat="density", alpha=0.4, edgecolor=None)
    sns.histplot(test_normal_errors, bins=50, color="blue", label="Test (Normal)", stat="density", alpha=0.4, edgecolor=None)
    sns.histplot(test_anomaly_errors, bins=50, color="red", label="Test (Anomaly)", stat="density", alpha=0.4, edgecolor=None)

    plt.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.5f})")

    tp = np.sum(test_anomaly_errors > threshold)
    fn = np.sum(test_anomaly_errors <= threshold)
    tn = np.sum(test_normal_errors <= threshold)
    fp = np.sum(test_normal_errors > threshold)

    plt.title("Anomaly Score Distribution", fontsize=14, pad=15)
    plt.xlabel("Reconstruction Error / Score")
    plt.ylabel("Density")
    plt.legend()

    stats_text = f"TP: {tp} | FN: {fn}\nTN: {tn} | FP: {fp}"
    plt.gca().text(
        0.95,
        0.5,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    plt.tight_layout()
    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
    plt.close()
    return final_path


def plot_confusion_matrix(cm, target_names, precision, recall, f1, specificity=None, auc_roc=None, auc_pr=None, save_path=None, model_name=None):
    """Plots a confusion matrix annotated with key performance metrics."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=target_names, yticklabels=target_names)

    title_str = f"Confusion Matrix\nPrecision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"
    if specificity is not None:
        title_str += f"\nSpec: {specificity:.3f} | AUC-ROC: {auc_roc:.3f} | AUC-PR: {auc_pr:.3f}"

    plt.title(title_str)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
    plt.close()
    return final_path


def plot_anomaly_comparison(raw_images, model_outputs, errors, indices, title, save_path, model_name=None, view_mode: str = None, error_maps=None):
    """
    Creates a standardized three-column visualization.

    Overlay mode:
    1. Original image
    2. Original image with anomaly overlay
    3. Error heatmap

    Reconstruction mode:
    1. Original image
    2. Reconstructed image
    3. Error heatmap
    """
    indices = list(indices)
    if len(indices) == 0:
        return None

    resolved_mode = resolve_visualization_mode(model_name=model_name, view_mode=view_mode)
    n_images = min(10, len(indices))
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 4 * n_images), squeeze=False)
    fig.suptitle(title, fontsize=16)

    for row_idx, sample_idx in enumerate(indices[:n_images]):
        sample_error_map = None if error_maps is None else error_maps[sample_idx]
        _render_comparison_row(
            axes[row_idx],
            raw_images[sample_idx],
            model_outputs[sample_idx],
            _extract_score(errors, sample_idx),
            resolved_mode,
            error_map=sample_error_map,
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
    plt.close(fig)
    return final_path


def plot_prediction_sample(raw_image, model_output, score: float, save_path: str, model_name: str = None, view_mode: str = None, error_map=None, title: str = None):
    """
    Renders a single standardized prediction sample using the same layout
    as the multi-sample anomaly comparison plots.
    """
    resolved_mode = resolve_visualization_mode(model_name=model_name, view_mode=view_mode)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)

    if title:
        fig.suptitle(title, fontsize=14)

    _render_comparison_row(axes[0], raw_image, model_output, float(score), resolved_mode, error_map=error_map)

    if title:
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    else:
        plt.tight_layout()
    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
    plt.close(fig)
    return final_path


def _batched_model_outputs(model, feature_tensor: torch.Tensor, view_mode: str):
    """
    Runs model inference in mini-batches to avoid moving the entire feature tensor
    to the GPU at once during analysis plotting.
    """
    outputs = []
    error_maps = []
    batch_size = config.BATCH_SIZE

    with torch.no_grad():
        for start in range(0, feature_tensor.shape[0], batch_size):
            end = start + batch_size
            feature_batch = feature_tensor[start:end].to(config.DEVICE, non_blocking=True)
            output_batch = model(feature_batch)

            if view_mode == FEATURE_OVERLAY_MODE:
                diff = (feature_batch - output_batch) ** 2
                if diff.ndim == 2:
                    spatial_error = diff.view(diff.shape[0], 1, 24, 32)
                else:
                    spatial_error = torch.mean(diff, dim=1, keepdim=True)

                heatmap_batch = F.interpolate(
                    spatial_error,
                    size=feature_tensor.shape[-2:] if feature_tensor.ndim >= 4 else (config.IMG_SIZE, config.IMG_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                outputs.append(heatmap_batch.cpu())
                error_maps.append(heatmap_batch.cpu())
            else:
                outputs.append(output_batch.cpu())
                error_maps.append(torch.abs(feature_tensor[start:end] - output_batch.cpu()))

    return torch.cat(outputs), torch.cat(error_maps)


def generate_anomaly_analysis(model, feature_tensor, raw_images_tensor, all_errors, y_true, model_name, view_mode: str = None):
    """
    Generates best-hit and worst-miss visualizations with a standardized layout.

    For feature-space models, this builds anomaly overlays from spatialized feature errors.
    For reconstruction models, this shows the reconstructed image and pixel error heatmap.
    """
    resolved_mode = resolve_visualization_mode(model_name=model_name, view_mode=view_mode)

    model_outputs, error_maps = _batched_model_outputs(model, feature_tensor, resolved_mode)

    if resolved_mode == FEATURE_OVERLAY_MODE:
        if list(model_outputs.shape[-2:]) != list(raw_images_tensor.shape[-2:]):
            model_outputs = F.interpolate(
                model_outputs,
                size=raw_images_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            error_maps = F.interpolate(
                error_maps,
                size=raw_images_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        visualization_data = model_outputs.numpy()
        error_maps = error_maps.numpy()
    else:
        visualization_data = model_outputs.numpy()
        if raw_images_tensor.shape == model_outputs.shape:
            error_maps = error_maps.numpy()
        else:
            error_maps = None

    raw_images = raw_images_tensor.detach().cpu().numpy()
    all_errors = np.asarray(all_errors)
    y_true = np.asarray(y_true)
    anomaly_mask = y_true == 1
    real_indices = np.where(anomaly_mask)[0]
    anomaly_scores = all_errors[anomaly_mask]

    top_hits = real_indices[np.argsort(anomaly_scores)[-10:][::-1]]
    top_misses = real_indices[np.argsort(anomaly_scores)[:10]]

    plot_anomaly_comparison(
        raw_images,
        visualization_data,
        all_errors,
        top_hits,
        "Top 10 Correctly Identified Anomalies",
        "heatmaps_best_hits.png",
        model_name=model_name,
        view_mode=resolved_mode,
        error_maps=error_maps,
    )
    plot_anomaly_comparison(
        raw_images,
        visualization_data,
        all_errors,
        top_misses,
        "Top 10 Missed Anomalies (False Negatives)",
        "heatmaps_worst_misses.png",
        model_name=model_name,
        view_mode=resolved_mode,
        error_maps=error_maps,
    )
