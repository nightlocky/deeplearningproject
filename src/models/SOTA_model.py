"""
PatchCore SOTA baseline with:
- PR / ROC curves
- Heatmaps for OCT anomaly visualization
"""

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import cv2

from torchvision import models
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
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
from src.helper.mlflow_helper import MLFlowTracker

MODEL_NAME = "patchcore"

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
        for imgs, labels in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(config.DEVICE)
            imgs3 = imgs.repeat(1, 3, 1, 1)

            _ = model(imgs3)

            f2 = features["layer2"]
            f3 = features["layer3"]

            f2 = F.interpolate(f2, size=f3.shape[2:], mode="bilinear")
            f = torch.cat([f2, f3], dim=1)
            f = F.avg_pool2d(f, 3, 1, 1)

            all_feats.append(f.cpu())
            all_labels.extend(labels.numpy())
            all_imgs.append(imgs.cpu())

    return torch.cat(all_feats), np.array(all_labels), torch.cat(all_imgs)

# ------------------------------
# Flatten patches
# ------------------------------
def flatten_patches(f):
    n, c, h, w = f.shape
    return f.permute(0,2,3,1).reshape(n*h*w, c).numpy()

# ------------------------------
# Score images
# ------------------------------
def score_images(f, knn, top_k=0.1):
    scores = []
    n, c, h, w = f.shape

    for i in tqdm(range(n), desc="Scoring"):
        patches = f[i].permute(1,2,0).reshape(h*w, c).numpy()
        dists, _ = knn.kneighbors(patches)
        d = dists[:,0]

        k = max(1, int(len(d)*top_k))
        scores.append(np.mean(np.sort(d)[-k:]))

    return np.array(scores)

# ------------------------------
# Heatmaps
# ------------------------------
def build_maps(f, knn):
    maps = []
    n, c, h, w = f.shape

    for i in range(n):
        patches = f[i].permute(1,2,0).reshape(h*w, c).numpy()
        dists, _ = knn.kneighbors(patches)
        maps.append(dists[:,0].reshape(h,w))

    return np.array(maps)

def get_output_paths():
    model_graph_dir = os.path.join(config.GRAPHS_DIR, MODEL_NAME)
    samples_dir = os.path.join(model_graph_dir, "samples")
    os.makedirs(model_graph_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    return {
        "model_graph_dir": model_graph_dir,
        "samples_dir": samples_dir,
    }


def save_heatmaps(images, maps, samples_dir):

    for i in range(min(10, len(images))):
        img = images[i][0].numpy()
        amap = maps[i]

        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        amap = cv2.resize(amap, (img.shape[1], img.shape[0]))

        heat = cv2.applyColorMap((amap*255).astype(np.uint8), cv2.COLORMAP_JET)
        img_rgb = np.stack([img]*3, axis=-1)

        overlay = (0.6*img_rgb + 0.4*heat).astype(np.uint8)
        out_path = os.path.join(samples_dir, f"heatmap_sample_{i}.png")
        cv2.imwrite(out_path, overlay)


def save_ranked_case_heatmaps(images, maps, scores, y_true, preds, threshold, samples_dir, top_n=5):
    ranked_dir = os.path.join(samples_dir, "best_worst_cases")
    os.makedirs(ranked_dir, exist_ok=True)

    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)

    anomaly_conf = scores - threshold
    normal_conf = threshold - scores

    tp = np.where((y_true == 1) & (preds == 1))[0]
    tn = np.where((y_true == 0) & (preds == 0))[0]
    fp = np.where((y_true == 0) & (preds == 1))[0]
    fn = np.where((y_true == 1) & (preds == 0))[0]

    cases = [
        ("best_hits_anomaly", tp, anomaly_conf, True),
        ("worst_hits_anomaly", tp, anomaly_conf, False),
        ("best_hits_normal", tn, normal_conf, True),
        ("worst_hits_normal", tn, normal_conf, False),
        ("best_misses_normal", fp, anomaly_conf, False),
        ("worst_misses_normal", fp, anomaly_conf, True),
        ("best_misses_anomaly", fn, normal_conf, False),
        ("worst_misses_anomaly", fn, normal_conf, True),
    ]

    for case_name, idxs, conf_values, descending in cases:
        case_dir = os.path.join(ranked_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)

        if len(idxs) == 0:
            continue

        ranked = idxs[np.argsort(conf_values[idxs])]
        if descending:
            ranked = ranked[::-1]

        for rank, i in enumerate(ranked[:top_n]):
            img = images[i][0].numpy()
            amap = maps[i]

            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            amap = cv2.resize(amap, (img.shape[1], img.shape[0]))
            heat = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img_rgb = np.stack([img] * 3, axis=-1)
            overlay = (0.6 * img_rgb + 0.4 * heat).astype(np.uint8)

            out_path = os.path.join(
                case_dir,
                f"{rank:02d}_idx{i}_score{scores[i]:.4f}_true{int(y_true[i])}_pred{int(preds[i])}.png",
            )
            cv2.imwrite(out_path, overlay)

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":

    paths = get_output_paths()

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

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(config.DEVICE)
    model.eval()

    model.layer2.register_forward_hook(hook("layer2"))
    model.layer3.register_forward_hook(hook("layer3"))

    print("Extracting features...")
    train_f, _, _ = extract_features(train_loader, model)
    test_f, test_labels, test_imgs = extract_features(test_loader, model)

    y_true = np.array([0 if l == normal_idx else 1 for l in test_labels])

    print("Building memory bank...")
    mem = flatten_patches(train_f)
    mem = mem[np.random.choice(len(mem), int(0.1*len(mem)), replace=False)]

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(mem)

    print("Scoring...")
    train_scores = score_images(train_f, knn)
    test_scores = score_images(test_f, knn)

    # PatchCore is non-parametric (no gradient training loop), so we log a
    # score-proxy curve to keep visualization outputs consistent with other models.
    plot_loss(train_scores.tolist(), "loss.png", MODEL_NAME)

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
        ["Normal","Anomaly"],
        precision,
        recall,
        f1,
        0,
        auc_roc,
        auc_pr,
        "cm.png",
        MODEL_NAME
    )

    plot_error_distribution(
        train_scores,
        test_scores[y_true==0],
        test_scores[y_true==1],
        thresh,
        "dist.png",
        MODEL_NAME
    )

    print("Generating heatmaps...")
    maps = build_maps(test_f, knn)
    save_heatmaps(test_imgs, maps, paths["samples_dir"])
    save_ranked_case_heatmaps(test_imgs, maps, test_scores, y_true, preds, thresh, paths["samples_dir"], top_n=5)