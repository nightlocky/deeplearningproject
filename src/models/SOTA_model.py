"""
Module: dra_patchcore_style.py
Description: DRA-style anomaly detection model aligned with PatchCore pipeline.
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src import config
from src.dataLoader.data_loader import dataloader
from src.helper.visualization_helper import (
    plot_loss,
    plot_error_distribution,
    plot_confusion_matrix,
    generate_anomaly_analysis,
)
from src.helper.mlflow_helper import MLFlowTracker


# ---------------------------------------------------------
# 1. Model
# ---------------------------------------------------------
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


class NormalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1)).squeeze(1)


class DRAModel(nn.Module):
    def __init__(self, top_k=0.1):
        super().__init__()
        self.backbone = Backbone()
        self.seen = Head()
        self.pseudo = Head()
        self.residual = Head()
        self.normal = NormalHead()
        self.top_k = top_k

    def topk(self, x):
        x = x.view(x.shape[0], -1)
        k = max(1, int(x.shape[1] * self.top_k))
        return torch.topk(x, k, dim=1)[0].mean(1)

    def forward(self, x, ref_feat):
        feat = self.backbone(x)

        seen = self.topk(self.seen(feat))
        pseudo = self.topk(self.pseudo(feat))
        residual = self.topk(self.residual(feat - ref_feat))
        normal = self.normal(feat)

        return seen + pseudo + residual - normal


# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------
def build_pseudo(x):
    x = x.clone()
    b, c, h, w = x.shape
    for i in range(b):
        ph, pw = h // 4, w // 4
        y, x_ = np.random.randint(0, h-ph), np.random.randint(0, w-pw)
        x[i, :, y:y+ph, x_:x_+pw] = torch.randn_like(x[i, :, y:y+ph, x_:x_+pw])
    return x


def compute_reference(model, loader, normal_idx):
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(config.DEVICE)
            labels = (labels == normal_idx)
            if labels.sum() == 0:
                continue
            f = model.backbone(imgs[labels])
            feats.append(f.mean(0, keepdim=True))
    if not feats:
        raise RuntimeError("No normal samples found to compute reference features.")
    return torch.mean(torch.cat(feats), dim=0, keepdim=True)


def split_train_val_loader(train_loader, val_ratio=0.2, seed=42):
    dataset = train_loader.dataset
    n_samples = len(dataset)
    if n_samples < 2:
        raise ValueError("Need at least 2 training samples to split train/val.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    val_size = max(1, int(n_samples * val_ratio))
    val_size = min(val_size, n_samples - 1)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_split_loader = DataLoader(
        train_subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if train_loader.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if train_loader.num_workers > 0 else False,
    )

    return train_split_loader, val_loader


def collect_val_with_pseudo_scores(model, loader, ref_feat):
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(config.DEVICE)

            normal_scores = torch.sigmoid(model(imgs, ref_feat)).cpu().numpy()
            pseudo_imgs = build_pseudo(imgs)
            pseudo_scores = torch.sigmoid(model(pseudo_imgs, ref_feat)).cpu().numpy()

            scores.extend(normal_scores)
            labels.extend(np.zeros(len(normal_scores), dtype=np.int32))
            scores.extend(pseudo_scores)
            labels.extend(np.ones(len(pseudo_scores), dtype=np.int32))

    return np.array(scores), np.array(labels)


def collect_test_scores(model, loader, ref_feat, test_normal_idx):
    scores, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(config.DEVICE)
            s = torch.sigmoid(model(imgs, ref_feat)).cpu().numpy()
            scores.extend(s)
            labels_all.extend(labels.numpy())

    y_true = np.array([0 if l == test_normal_idx else 1 for l in labels_all])
    return np.array(scores), y_true


# =========================================================
# EXECUTION
# =========================================================
if __name__ == "__main__":

    train_loader, test_loader, normal_idx = dataloader(
        train_path=config.TRAIN_PATH,
        test_path=config.TEST_PATH,
        n_train_normal=config.N_TRAIN_NORMAL,
        n_test_normal=config.N_TEST_NORMAL,
        n_test_anomaly_per_class=config.N_TEST_ANOMALY_PER_CLASS,
        img_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    if hasattr(test_loader.dataset, "dataset") and hasattr(test_loader.dataset.dataset, "class_to_idx"):
        test_normal_idx = test_loader.dataset.dataset.class_to_idx["NORMAL"]
    elif hasattr(test_loader.dataset, "class_to_idx"):
        test_normal_idx = test_loader.dataset.class_to_idx["NORMAL"]
    else:
        test_normal_idx = normal_idx

    train_loader, val_loader = split_train_val_loader(train_loader, val_ratio=0.2, seed=42)

    model = DRAModel().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    tracker = MLFlowTracker(experiment_name="DRA_SOTA")

    with tracker:
        train_losses = []
        val_auc_pr_history = []
        max_epochs = 50
        patience = 8
        best_val_auc_pr = -1.0
        no_improve_count = 0
        best_state = copy.deepcopy(model.state_dict())

        print("\nTraining DRA...")
        for epoch in range(max_epochs):
            model.train()
            batch_losses = []

            for imgs, labels in tqdm(train_loader):
                imgs = imgs.to(config.DEVICE)
                labels = (labels != normal_idx).float().to(config.DEVICE)

                normal_imgs = imgs[labels == 0]
                if len(normal_imgs) == 0:
                    continue

                ref_feat = model.backbone(normal_imgs).mean(0, keepdim=True)

                scores = model(imgs, ref_feat)
                loss_main = F.binary_cross_entropy_with_logits(scores, labels)

                pseudo_imgs = build_pseudo(normal_imgs)
                pseudo_scores = model(pseudo_imgs, ref_feat)
                loss_pseudo = F.binary_cross_entropy_with_logits(
                    pseudo_scores,
                    torch.ones(len(pseudo_imgs)).to(config.DEVICE)
                )

                loss = loss_main + loss_pseudo

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            avg_loss = np.mean(batch_losses) if batch_losses else 0
            train_losses.append(avg_loss)
            ref_feat_val = compute_reference(model, train_loader, normal_idx)
            val_scores, val_labels = collect_val_with_pseudo_scores(model, val_loader, ref_feat_val)
            val_auc_pr = average_precision_score(val_labels, val_scores)
            val_auc_pr_history.append(val_auc_pr)

            print(f"Epoch {epoch+1}: loss={avg_loss:.4f} | val_auc_pr={val_auc_pr:.4f}")

            if val_auc_pr > best_val_auc_pr:
                best_val_auc_pr = val_auc_pr
                best_state = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch+1} (best val_auc_pr={best_val_auc_pr:.4f}).")
                    break

        model.load_state_dict(best_state)

        # ---------------------------------------------------------
        # Evaluation without test leakage
        # ---------------------------------------------------------
        model.eval()
        ref_feat = compute_reference(model, train_loader, normal_idx)

        val_s, val_y = collect_val_with_pseudo_scores(model, val_loader, ref_feat)
        test_s, test_y = collect_test_scores(model, test_loader, ref_feat, test_normal_idx)
        print(f"test_y bincount [normal(0), anomaly(1)]: {np.bincount(test_y, minlength=2)}")

        best_t, best_f1 = 0, 0
        for t in np.linspace(min(val_s), max(val_s), 500):
            preds = (val_s > t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(val_y, preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        test_preds = (np.array(test_s) > best_t).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(test_y, test_preds, average='binary', zero_division=0)
        cm = confusion_matrix(test_y, test_preds)

        auc_roc = roc_auc_score(test_y, test_s)
        auc_pr = average_precision_score(test_y, test_s)
        report = classification_report(test_y, test_preds, output_dict=True, zero_division=0)

        print("\nFINAL RESULTS")
        print(f"Support class 0: {int(report['0']['support'])}")
        print(f"Support class 1: {int(report['1']['support'])}")
        print(classification_report(test_y, test_preds, zero_division=0))
        print(f"Best threshold from val: {best_t:.4f}")
        print(f"AUC ROC: {auc_roc:.4f} | AUC PR: {auc_pr:.4f}")