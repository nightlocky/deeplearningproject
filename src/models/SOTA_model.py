"""
PatchCore-style SOTA baseline with GPU-accelerated patch scoring.

Key behavior:
- pretrained ResNet18 feature extractor
- layer2/layer3 PatchCore-style features
- memory bank built from normal training patches
- anomaly score from nearest-neighbor patch distance
- heatmap overlays saved under graphs/SOTA_model/
"""

import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src import config
from src.dataLoader.data_loader import dataloader
from src.helper.visualization_helper import (
    plot_confusion_matrix,
    plot_error_distribution,
)

MODEL_NAME = "SOTA_model"
MEMORY_BANK_RATIO = 0.01
PATCH_BATCH_SIZE = 8192


features = {}


def hook(name):
    def fn(module, input, output):
        features[name] = output.detach()

    return fn


def get_output_paths():
    model_graph_dir = os.path.join(config.GRAPHS_DIR, MODEL_NAME)
    samples_dir = os.path.join(model_graph_dir, "samples")
    os.makedirs(model_graph_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    return {
        "model_graph_dir": model_graph_dir,
        "samples_dir": samples_dir,
    }


def extract_features(loader, model):
    all_feats, all_labels, all_imgs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            imgs3 = imgs.repeat(1, 3, 1, 1)

            _ = model(imgs3)

            f2 = features["layer2"]
            f3 = features["layer3"]

            f2 = F.interpolate(f2, size=f3.shape[2:], mode="bilinear", align_corners=False)
            f = torch.cat([f2, f3], dim=1)

            # Downsample the spatial grid to keep the PatchCore baseline fast.
            f = F.avg_pool2d(f, 4, 4)

            all_feats.append(f.cpu())
            all_labels.extend(labels.numpy())
            all_imgs.append(imgs.cpu())

    return torch.cat(all_feats), np.array(all_labels), torch.cat(all_imgs)


def flatten_all_patches(feature_maps):
    n, c, h, w = feature_maps.shape
    patches = feature_maps.permute(0, 2, 3, 1).reshape(n * h * w, c).contiguous()
    return patches, (n, h, w)


def build_memory_bank(train_features):
    patches, _ = flatten_all_patches(train_features)
    sample_count = max(1, int(MEMORY_BANK_RATIO * len(patches)))
    sampled_idx = torch.randperm(len(patches))[:sample_count]
    return patches[sampled_idx].to(config.DEVICE, non_blocking=True)


def compute_patch_distances(feature_maps, memory_bank, patch_batch_size=PATCH_BATCH_SIZE):
    patches, (n, h, w) = flatten_all_patches(feature_maps)
    nearest_chunks = []

    print("Running GPU KNN (batched)...")
    with torch.no_grad():
        for start in tqdm(range(0, len(patches), patch_batch_size), desc="Patch distance batches"):
            patch_batch = patches[start : start + patch_batch_size].to(config.DEVICE, non_blocking=True)
            dists = torch.cdist(patch_batch, memory_bank)
            nearest_chunks.append(dists.amin(dim=1).cpu())

    patch_dists = torch.cat(nearest_chunks).reshape(n, h * w)
    anomaly_maps = patch_dists.reshape(n, h, w).numpy()
    image_scores = patch_dists.mean(dim=1).numpy()
    return image_scores, anomaly_maps


def save_heatmaps(images, maps, samples_dir, max_images=10):
    for i in tqdm(range(min(max_images, len(images))), desc="Saving heatmaps"):
        img = images[i][0].numpy()
        amap = maps[i]

        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        amap = cv2.resize(amap, (img.shape[1], img.shape[0]))

        heat = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img_rgb = (np.stack([img] * 3, axis=-1) * 255).astype(np.uint8)
        overlay = cv2.addWeighted(img_rgb, 0.6, heat, 0.4, 0)

        cv2.imwrite(os.path.join(samples_dir, f"heatmap_{i}.png"), overlay)


if __name__ == "__main__":
    start_total = time.time()
    paths = get_output_paths()

    print("Loading data...")
    train_loader, test_loader, normal_idx = dataloader(
        train_path=config.TRAIN_PATH,
        test_path=config.TEST_PATH,
        img_size=config.IMG_SIZE,
        n_train_normal=config.N_TRAIN_NORMAL,
        n_test_normal=config.N_TEST_NORMAL,
        n_test_anomaly_per_class=config.N_TEST_ANOMALY_PER_CLASS,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    ).to(config.DEVICE)
    model.eval()

    model.layer2.register_forward_hook(hook("layer2"))
    model.layer3.register_forward_hook(hook("layer3"))

    print("Extracting features...")
    train_f, _, _ = extract_features(train_loader, model)
    test_f, test_labels, test_imgs = extract_features(test_loader, model)
    y_true = np.array([0 if label == normal_idx else 1 for label in test_labels])

    print("Building memory bank...")
    memory_bank = build_memory_bank(train_f)

    print("Scoring train-normal set...")
    train_scores, _ = compute_patch_distances(train_f, memory_bank)

    print("Scoring test set...")
    test_scores, test_maps = compute_patch_distances(test_f, memory_bank)

    # PatchCore uses the normal training distribution to set the threshold.
    thresh = np.percentile(train_scores, 95)
    preds = (test_scores > thresh).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    auc_roc = roc_auc_score(y_true, test_scores)
    auc_pr = average_precision_score(y_true, test_scores)

    print("\n=== RESULTS ===")
    print(classification_report(y_true, preds))
    print(f"AUC ROC: {auc_roc:.4f}")
    print(f"AUC PR: {auc_pr:.4f}")

    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-8)
    print(f"Specificity: {specificity:.4f}")

    plot_confusion_matrix(
        cm,
        ["Normal", "Anomaly"],
        precision,
        recall,
        f1,
        specificity=specificity,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        save_path=os.path.join(paths["model_graph_dir"], "test_confusion_matrix.png"),
        model_name=MODEL_NAME,
    )

    plot_error_distribution(
        train_scores,
        test_scores[y_true == 0],
        test_scores[y_true == 1],
        thresh,
        os.path.join(paths["model_graph_dir"], "test_anomaly_score_distribution.png"),
        MODEL_NAME,
    )

    print("Generating heatmaps...")
    save_heatmaps(test_imgs, test_maps, paths["samples_dir"])

    print(f"\nTOTAL TIME: {time.time() - start_total:.2f} sec")
