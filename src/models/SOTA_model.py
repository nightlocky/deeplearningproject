"""
ULTRA FAST PatchCore (tqdm everywhere, <5 mins)

Key optimizations:
- Memory bank = 1%
- Feature downsampling
- PCA compression
- Chunked KNN (with tqdm)
- tqdm for ALL stages
"""

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import time

from torchvision import models
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src import config
from src.dataLoader.data_loader import dataloader
from src.helper.visualization_helper import (
    plot_loss,
    plot_error_distribution,
    plot_confusion_matrix,
)

MODEL_NAME = "patchcore_ultrafast"

# ------------------------------
# Feature hooks
# ------------------------------
features = {}

def hook(name):
    def fn(module, input, output):
        features[name] = output.detach()
    return fn

# ------------------------------
# Feature extraction
# ------------------------------
def extract_features(loader, model):
    all_feats, all_labels, all_imgs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(config.DEVICE)
            imgs3 = imgs.repeat(1, 3, 1, 1)

            _ = model(imgs3)

            f2 = features["layer2"]
            f3 = features["layer3"]

            f2 = F.interpolate(f2, size=f3.shape[2:], mode="bilinear")
            f = torch.cat([f2, f3], dim=1)

            # 🔥 MASSIVE SPEED BOOST
            f = F.avg_pool2d(f, 4, 4)

            all_feats.append(f.cpu())
            all_labels.extend(labels.numpy())
            all_imgs.append(imgs.cpu())

    return torch.cat(all_feats), np.array(all_labels), torch.cat(all_imgs)

# ------------------------------
# Flatten patches
# ------------------------------
def flatten_all_patches(f):
    n, c, h, w = f.shape
    return f.permute(0,2,3,1).reshape(n*h*w, c).numpy(), (n,h,w)

# ------------------------------
# PCA with tqdm
# ------------------------------
def pca_transform_with_progress(pca, data, batch_size=50000):
    out = []
    for i in tqdm(range(0, len(data), batch_size), desc="PCA transform"):
        out.append(pca.transform(data[i:i+batch_size]))
    return np.vstack(out)

# ------------------------------
# Chunked KNN (tqdm)
# ------------------------------
def knn_with_progress(knn, patches, batch_size=50000):
    all_dists = []
    for i in tqdm(range(0, len(patches), batch_size), desc="KNN scoring"):
        batch = patches[i:i+batch_size]
        dists, _ = knn.kneighbors(batch)
        all_dists.append(dists[:,0])
    return np.concatenate(all_dists)

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":

    start_total = time.time()

    print("Loading data...")
    train_loader, test_loader, normal_idx = dataloader(
        train_path=config.TRAIN_PATH,
        test_path=config.TEST_PATH,
        img_size=config.IMG_SIZE,
        n_train_normal=config.N_TRAIN_NORMAL,
        n_test_normal=config.N_TEST_NORMAL,
        n_test_anomaly_per_class=config.N_TEST_ANOMALY_PER_CLASS,
        batch_size=64,
        num_workers=16,
    )

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    ).to(config.DEVICE)
    model.eval()

    model.layer2.register_forward_hook(hook("layer2"))
    model.layer3.register_forward_hook(hook("layer3"))

    # ------------------------------
    # Feature extraction
    # ------------------------------
    print("Extracting features...")
    train_f, _, _ = extract_features(train_loader, model)
    test_f, test_labels, test_imgs = extract_features(test_loader, model)

    y_true = np.array([0 if l == normal_idx else 1 for l in test_labels])

    # ------------------------------
    # Memory bank
    # ------------------------------
    print("Building memory bank...")
    mem, _ = flatten_all_patches(train_f)

    print("Subsampling memory...")
    mem = mem[np.random.choice(len(mem), int(0.01 * len(mem)), replace=False)]

    # ------------------------------
    # PCA
    # ------------------------------
    print("Applying PCA...")
    pca = PCA(n_components=64)
    mem = pca.fit_transform(mem)

    # ------------------------------
    # KNN
    # ------------------------------
    print("Fitting KNN...")
    knn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    knn.fit(mem)

    # ------------------------------
    # TEST SCORING
    # ------------------------------
    print("Preparing test patches...")
    test_patches, (n,h,w) = flatten_all_patches(test_f)

    print("Applying PCA to test...")
    test_patches = pca_transform_with_progress(pca, test_patches)

    print("Running KNN...")
    d = knn_with_progress(knn, test_patches)

    d = d.reshape(n, h*w)

    print("Aggregating scores...")
    scores = []
    for i in tqdm(range(n), desc="Aggregating"):
        scores.append(np.mean(d[i]))
    scores = np.array(scores)

    # ------------------------------
    # Metrics
    # ------------------------------
    thresh = np.percentile(scores, 95)
    preds = (scores > thresh).astype(int)

    print("\n=== RESULTS ===")
    print(classification_report(y_true, preds))

    auc_roc = roc_auc_score(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)

    print(f"AUC ROC: {auc_roc:.4f}")
    print(f"AUC PR: {auc_pr:.4f}")

    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-8)
    print(f"Specificity: {specificity:.4f}")

    plot_confusion_matrix(
        cm,
        ["Normal", "Anomaly"],
        0,0,0,0,
        auc_roc,
        auc_pr,
        "cm.png",
        MODEL_NAME
    )

    plot_error_distribution(
        scores,
        scores[y_true == 0],
        scores[y_true == 1],
        thresh,
        "dist.png",
        MODEL_NAME
    )

    # ------------------------------
    # Heatmaps
    # ------------------------------
    print("Generating heatmaps...")
    maps = d.reshape(n, h, w)

    os.makedirs(config.GRAPHS_DIR, exist_ok=True)

    for i in tqdm(range(min(10, len(test_imgs))), desc="Saving heatmaps"):
        img = test_imgs[i][0].numpy()
        amap = maps[i]

        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        amap = cv2.resize(amap, (img.shape[1], img.shape[0]))

        heat = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img_rgb = np.stack([img]*3, axis=-1)

        overlay = (0.6 * img_rgb + 0.4 * heat).astype(np.uint8)

        cv2.imwrite(
            os.path.join(config.GRAPHS_DIR, f"heatmap_{i}.png"),
            overlay
        )

    print(f"\nTOTAL TIME: {time.time() - start_total:.2f} sec")