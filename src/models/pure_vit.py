"""
Module: pure_vit.py
Description: Evaluates anomaly detection by computing the centroid distance of 
features extracted using a pretrained Vision Transformer (ViT).
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from src import config
from src.dataLoader.data_loader import dataloader
from src.helper.visualization_helper import plot_loss, plot_error_distribution, plot_confusion_matrix, plot_anomaly_comparison
from src.helper.mlflow_helper import MLFlowTracker

# ---------------------------------------------------------
# 1. Configuration Constants
# ---------------------------------------------------------
# =========================================================
# EXECUTION BLOCK (Protected for Multiprocessing on RunPod/Windows)
# =========================================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading datasets...")
    # NOTE: Your `dataloader` outputs 1-channel grayscale images. 
    # ViT expects 3 channels. We will expand the channels inside the feature extraction loops.
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

    # ---------------------------------------------------------
    # 3. Model: Vision Transformer (Feature Extractor)
    # ---------------------------------------------------------
    print(f"Loading ViT-B/16 onto {config.DEVICE}...")
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity() 
    vit = vit.to(config.DEVICE)
    vit.eval() 

    # ---------------------------------------------------------
    # 4. MLflow Tracking & Logic
    # ---------------------------------------------------------
    tracker = MLFlowTracker(experiment_name="ViT_Centroid_Anomaly")

    with tracker as run:
        tracker.log_params({
            "backbone": "vit_b_16_pretrained",
            "strategy": "Centroid_Distance",
            "img_size": config.IMG_SIZE,
            "batch_size": config.BATCH_SIZE
        })

        print("\nExtracting Training Features (Normal data only)...")
        train_features = []
        with torch.no_grad():
            for images, _ in train_loader:
                # ViT needs 3 channels, grayscale loader returns 1.
                images_3c = images.repeat(1, 3, 1, 1).to(config.DEVICE)
                feats = vit(images_3c)
                train_features.append(feats.cpu().numpy())
        
        train_features = np.concatenate(train_features)
        
        # Calculate the Centroid (mean embedding of all normal training data)
        normal_center = np.mean(train_features, axis=0)
        
        # Calculate training distances to set a threshold
        train_distances = np.linalg.norm(train_features - normal_center, axis=1)
        
        # Set threshold at the 95th percentile of normal training distances
        # Expect a 5% false positive rate on perfectly normal data
        THRESHOLD = np.percentile(train_distances, 95)
        tracker.log_metrics("distance_threshold", THRESHOLD)
        print(f"Normal Center calculated. Anomaly Threshold set to: {THRESHOLD:.4f}")

        print("\nEvaluating Test Set...")
        test_distances = []
        test_labels_raw = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images_3c = images.repeat(1, 3, 1, 1).to(config.DEVICE)
                feats = vit(images_3c).cpu().numpy()
                dist = np.linalg.norm(feats - normal_center, axis=1)
                test_distances.extend(dist)
                test_labels_raw.extend(labels.numpy())
                
        test_distances = np.array(test_distances)
        
        y_true = [0 if l == normal_idx else 1 for l in test_labels_raw]
        y_pred = [1 if d > THRESHOLD else 0 for d in test_distances]

        # Calculate standard classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        # Calculate Specificity (True Negative Rate)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate AUC metrics using the continuous centroid distances
        auc_roc = roc_auc_score(y_true, test_distances)
        auc_pr = average_precision_score(y_true, test_distances)

        tracker.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr
        })

        print("\n" + "="*30)
        print("FINAL PERFORMANCE (ViT CENTROID DISTANCE)")
        target_names = ['Healthy (Normal)', 'Pathology (Anomaly)']
        print(classification_report(y_true, y_pred, target_names=target_names))

        # ---------------------------------------------------------
        # 5. Visualization & Artifacts
        # ---------------------------------------------------------
        MODEL_NAME = "vit_centroid"
        
        cm_path = plot_confusion_matrix(
            cm=cm, 
            target_names=target_names, 
            precision=precision,
            recall=recall,
            f1=f1,
            specificity=specificity,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            save_path="vit_centroid_cm.png", 
            model_name=MODEL_NAME
        )
        tracker.log_artifact(cm_path)
        
        test_normal_dists = test_distances[np.array(y_true) == 0]
        test_anomaly_dists = test_distances[np.array(y_true) == 1]
        
        dist_path = plot_error_distribution(
            train_errors=train_distances, 
            test_normal_errors=test_normal_dists, 
            test_anomaly_errors=test_anomaly_dists, 
            threshold=THRESHOLD, 
            save_path="vit_distance_dist.png", 
            model_name=MODEL_NAME
        )
        tracker.log_artifact(dist_path)
        
        print(f"\nResults logged to MLflow under experiment: {tracker.experiment_name}")
