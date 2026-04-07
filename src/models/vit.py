<<<<<<< HEAD
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
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt

# Ensure the parent directory (src) and its data subdirectory are in the path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)

# Import custom helpers
from dataLoader.dataLoader import get_anomaly_dataloaders
from visualization_helper import plot_confusion_matrix_custom, plot_error_distribution
from mlflow_helper import MLFlowTracker

# ---------------------------------------------------------
# 1. Configuration & Data Loading
# ---------------------------------------------------------
TRAIN_PATH = os.path.join(project_root, 'data', 'OCT', 'train')
TEST_PATH = os.path.join(project_root, 'data', 'OCT', 'test')

IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading datasets...")
train_loader, test_loader, normal_idx = get_anomaly_dataloaders(
    train_path=TRAIN_PATH, 
    test_path=TEST_PATH, 
    img_size=IMG_SIZE,
    n_train_normal = 1000,
    n_test_normal = 300,
    n_test_anomaly=50,
    batch_size=BATCH_SIZE
)

# ---------------------------------------------------------
# 2. Model: Vision Transformer (Feature Extractor)
# ---------------------------------------------------------
print(f"Loading ViT-B/16 onto {DEVICE}...")
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
# Remove the classification head to get the raw 768-dim features
vit.heads = nn.Identity() 
vit = vit.to(DEVICE)
vit.eval() # We are not training the ViT, only extracting features

# ---------------------------------------------------------
# 3. MLflow Tracking & Logic
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="ViT_Centroid_Anomaly")

with tracker as run:
    tracker.log_params({
        "backbone": "vit_b_16_pretrained",
        "strategy": "Centroid_Distance",
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE
    })

    # --- Step A: Extract Training Features & Find the "Normal Center" ---
    print("\nExtracting Training Features (Normal data only)...")
    train_features = []
    with torch.no_grad():
        for imgs, _ in train_loader:
            feats = vit(imgs.to(DEVICE))
            train_features.append(feats.cpu().numpy())
    
    train_features = np.concatenate(train_features)
    
    # Calculate the Centroid (mean embedding of all normal data)
    normal_center = np.mean(train_features, axis=0)
    
    # Calculate training distances to set a threshold
    train_distances = np.linalg.norm(train_features - normal_center, axis=1)
    
    # Set threshold at the 95th percentile of normal training distances
    # This means we expect a 5% false positive rate on perfectly normal data
    THRESHOLD = np.percentile(train_distances, 95)
    tracker.log_metric("distance_threshold", THRESHOLD)
    print(f"Normal Center calculated. Anomaly Threshold set to: {THRESHOLD:.4f}")

    # --- Step B: Extract Test Features & Calculate Distances ---
    print("\nEvaluating Test Set...")
    test_distances = []
    test_labels_raw = []
    
    with torch.no_grad():
        for imgs, lbls in test_loader:
            feats = vit(imgs.to(DEVICE)).cpu().numpy()
            dist = np.linalg.norm(feats - normal_center, axis=1)
            test_distances.extend(dist)
            test_labels_raw.extend(lbls.numpy())
            
    test_distances = np.array(test_distances)
    
    # --- Step C: Predictions & Evaluation ---
    # Ground truth: 0 for Normal, 1 for Anomaly
    y_true = [0 if l == normal_idx else 1 for l in test_labels_raw]
    
    # Prediction: 1 (Anomaly) if distance > threshold, else 0 (Normal)
    y_pred = [1 if d > THRESHOLD else 0 for d in test_distances]

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    tracker.log_metrics({
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print("\n" + "="*30)
    print("FINAL PERFORMANCE (ViT CENTROID DISTANCE)")
    target_names = ['Healthy (Normal)', 'Pathology (Anomaly)']
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ---------------------------------------------------------
    # 4. Visualization & Artifacts
    # ---------------------------------------------------------
    MODEL_NAME = "vit_centroid"
    
    # 1. Confusion Matrix
    cm_path = plot_confusion_matrix_custom(cm, target_names, save_path="vit_centroid_cm.png", model_name=MODEL_NAME)
    tracker.log_artifact(cm_path)
    
    # 2. Distance Distribution
    test_normal_dists = test_distances[np.array(y_true) == 0]
    test_anomaly_dists = test_distances[np.array(y_true) == 1]
    
    # We pass the distances directly to your error distribution helper
    dist_path = plot_error_distribution(
        train_distances, 
        test_normal_dists, 
        test_anomaly_dists, 
        threshold=THRESHOLD, 
        save_path="vit_distance_dist.png", 
        model_name=MODEL_NAME
    )
    tracker.log_artifact(dist_path)
    
    print(f"\nResults logged to MLflow under experiment: {tracker.experiment_name}")
=======
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Ensure the parent directory (src) and its data subdirectory are in the path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)

# Import custom helpers
from dataLoader.dataLoader import get_anomaly_dataloaders
from helper.visualization_helper import plot_confusion_matrix_custom, plot_error_distribution
from helper.mlflow_helper import MLFlowTracker

# ---------------------------------------------------------
# 1. Configuration & Data Loading
# ---------------------------------------------------------
TRAIN_PATH = os.path.join(project_root, 'data', 'OCT', 'train')
TEST_PATH = os.path.join(project_root, 'data', 'OCT', 'test')

IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading datasets...")
train_loader, test_loader, normal_idx = get_anomaly_dataloaders(
    train_path=TRAIN_PATH, 
    test_path=TEST_PATH, 
    img_size=IMG_SIZE,
    n_train_normal = 1000,
    n_test_normal = 300,
    n_test_anomaly=50,
    batch_size=BATCH_SIZE
)

# ---------------------------------------------------------
# 2. Model: Vision Transformer (Feature Extractor)
# ---------------------------------------------------------
print(f"Loading ViT-B/16 onto {DEVICE}...")
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
# Remove the classification head to get the raw 768-dim features
vit.heads = nn.Identity() 
vit = vit.to(DEVICE)
vit.eval() # We are not training the ViT, only extracting features

# ---------------------------------------------------------
# 3. MLflow Tracking & Logic
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="ViT_Centroid_Anomaly")

with tracker as run:
    tracker.log_params({
        "backbone": "vit_b_16_pretrained",
        "strategy": "Centroid_Distance",
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE
    })

    # --- Step A: Extract Training Features & Find the "Normal Center" ---
    print("\nExtracting Training Features (Normal data only)...")
    train_features = []
    with torch.no_grad():
        for imgs, _ in train_loader:
            feats = vit(imgs.to(DEVICE))
            train_features.append(feats.cpu().numpy())
    
    train_features = np.concatenate(train_features)
    
    # Calculate the Centroid (mean embedding of all normal data)
    normal_center = np.mean(train_features, axis=0)
    
    # Calculate training distances to set a threshold
    train_distances = np.linalg.norm(train_features - normal_center, axis=1)
    
    # Set threshold at the 95th percentile of normal training distances
    # This means we expect a 5% false positive rate on perfectly normal data
    THRESHOLD = np.percentile(train_distances, 95)
    tracker.log_metric("distance_threshold", THRESHOLD)
    print(f"Normal Center calculated. Anomaly Threshold set to: {THRESHOLD:.4f}")

    # --- Step B: Extract Test Features & Calculate Distances ---
    print("\nEvaluating Test Set...")
    test_distances = []
    test_labels_raw = []
    
    with torch.no_grad():
        for imgs, lbls in test_loader:
            feats = vit(imgs.to(DEVICE)).cpu().numpy()
            dist = np.linalg.norm(feats - normal_center, axis=1)
            test_distances.extend(dist)
            test_labels_raw.extend(lbls.numpy())
            
    test_distances = np.array(test_distances)
    
    # --- Step C: Predictions & Evaluation ---
    # Ground truth: 0 for Normal, 1 for Anomaly
    y_true = [0 if l == normal_idx else 1 for l in test_labels_raw]
    
    # Prediction: 1 (Anomaly) if distance > threshold, else 0 (Normal)
    y_pred = [1 if d > THRESHOLD else 0 for d in test_distances]

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    tracker.log_metrics({
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print("\n" + "="*30)
    print("FINAL PERFORMANCE (ViT CENTROID DISTANCE)")
    target_names = ['Healthy (Normal)', 'Pathology (Anomaly)']
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ---------------------------------------------------------
    # 4. Visualization & Artifacts
    # ---------------------------------------------------------
    MODEL_NAME = "vit_centroid"
    
    # 1. Confusion Matrix
    cm_path = plot_confusion_matrix_custom(cm, target_names, save_path="vit_centroid_cm.png", model_name=MODEL_NAME)
    tracker.log_artifact(cm_path)
    
    # 2. Distance Distribution
    test_normal_dists = test_distances[np.array(y_true) == 0]
    test_anomaly_dists = test_distances[np.array(y_true) == 1]
    
    # We pass the distances directly to your error distribution helper
    dist_path = plot_error_distribution(
        train_distances, 
        test_normal_dists, 
        test_anomaly_dists, 
        threshold=THRESHOLD, 
        save_path="vit_distance_dist.png", 
        model_name=MODEL_NAME
    )
    tracker.log_artifact(dist_path)
    
    print(f"\nResults logged to MLflow under experiment: {tracker.experiment_name}")
>>>>>>> origin/edison
    plt.show()