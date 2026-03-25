import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Ensure the parent directory (src) and current directory are in the path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Import custom helpers
from visualization_helper import (
    plot_confusion_matrix_custom,
    plot_error_distribution # Useful for showing IF anomaly scores
)
from mlflow_helper import MLFlowTracker

# ---------------------------------------------------------
# 1. Configuration & Data Loading
# ---------------------------------------------------------
# Robust paths relative to project root
TRAIN_PATH = os.path.join(project_root, 'data', 'OCT', 'train')
TEST_PATH = os.path.join(project_root, 'data', 'OCT', 'test')
IMG_SIZE = 224 # ResNet default
BATCH_SIZE = 32

# ResNet expects specific normalization
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3), # ResNet expects 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
full_train_ds = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
full_test_ds = datasets.ImageFolder(root=TEST_PATH, transform=transform)

normal_idx = full_train_ds.class_to_idx['NORMAL']

# Train Set: 500 NORMAL images
train_normal_indices = [i for i, (_, label) in enumerate(full_train_ds.samples) if label == normal_idx]
train_subset = Subset(full_train_ds, train_normal_indices[:500]) 
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)

# Test Set: 250 NORMAL + 30 Anomalies
test_normal_indices = [i for i, (_, label) in enumerate(full_test_ds.samples) if label == normal_idx]
test_anomaly_indices = [i for i, (_, label) in enumerate(full_test_ds.samples) if label != normal_idx]

test_subset_indices = test_normal_indices[:250] + test_anomaly_indices[:30]
test_ds = Subset(full_test_ds, test_subset_indices)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset Ready!")
print(f"-> Using Pretrained CNN (ResNet18) for feature extraction.")
print(f"-> Extracting features from {len(train_subset)} Normal images.")

# ---------------------------------------------------------
# 2. Model: Pretrained CNN (ResNet18)
# ---------------------------------------------------------
# We use ResNet18 as a fixed feature extractor (no decoder needed)
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Remove the final classification layer to get the 512-dim feature vector
resnet.fc = nn.Identity() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)
resnet.eval()

# ---------------------------------------------------------
# 3. Feature Extraction
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="CNN_IsolationForest_Anomaly")

with tracker as run:
    tracker.log_params({
        "cnn_backbone": "resnet18_pretrained",
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "model_type": "CNN_IForest_NoDecoder"
    })

    def extract_features(loader):
        features = []
        labels = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                feat = resnet(imgs)
                features.append(feat.cpu().numpy())
                labels.append(lbls.numpy())
        return np.concatenate(features), np.concatenate(labels)

    print("\nExtracting features...")
    train_features, _ = extract_features(train_loader)
    test_features, test_labels_raw = extract_features(test_loader)
    
    y_true = [0 if l == normal_idx else 1 for l in test_labels_raw]

    # ---------------------------------------------------------
    # 4. Anomaly Detection: Isolation Forest
    # ---------------------------------------------------------
    print("Fitting Isolation Forest...")
    # contamination is the expected % of anomalies in the training set (here ~0)
    iso_forest = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iso_forest.fit(train_features)

    # Get anomaly scores (lower is more anomalous)
    scores = iso_forest.decision_function(test_features)
    # Get predictions: -1 for anomaly, 1 for normal
    test_preds_raw = iso_forest.predict(test_features)
    y_pred = [1 if p == -1 else 0 for p in test_preds_raw]

    # ---------------------------------------------------------
    # 5. Evaluation & Logging
    # ---------------------------------------------------------
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    tracker.log_metrics({
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print("\n" + "="*30)
    print("FINAL PERFORMANCE (CNN FEATURES + ISOLATION FOREST)")
    cm = confusion_matrix(y_true, y_pred)
    target_names = ['Healthy (Normal)', 'Pathology (Anomaly)']
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ---------------------------------------------------------
    # 6. Visualization & Artifacts
    # ---------------------------------------------------------
    MODEL_NAME = "cnn_iforest_pretrained"
    
    # Plot Confusion Matrix
    cm_path = plot_confusion_matrix_custom(cm, target_names, save_path="cnn_if_cm.png", model_name=MODEL_NAME)
    tracker.log_artifact(cm_path)
    
    # Plot Score Distribution (Using the error distribution helper for decision scores)
    train_scores = iso_forest.decision_function(train_features)
    test_normal_scores = scores[np.array(y_true) == 0]
    test_anomaly_scores = scores[np.array(y_true) == 1]
    
    # Note: Decision scores are higher for normal, lower for anomalies.
    # We negate them for the 'error' distribution plot to keep the 'higher is anomaly' visual.
    dist_path = plot_error_distribution(
        -train_scores, 
        -test_normal_scores, 
        -test_anomaly_scores, 
        threshold=-iso_forest.offset_, # offset_ is a scalar in newer sklearn
        save_path="score_dist.png", 
        model_name=MODEL_NAME
    )
    tracker.log_artifact(dist_path)
    
    print(f"\nResults logged to MLflow under experiment: {tracker.experiment_name}")
    plt.show()
