import os
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)


sys.path.append(parent_dir)
# Import your custom helpers
from dataLoader.dataLoader import get_anomaly_dataloaders
from visualization_helper import plot_loss, plot_error_distribution, plot_confusion_matrix_custom
from mlflow_helper import MLFlowTracker

def main():

    # ---------------------------------------------------------
    # 1. Configuration & Data Loading
    # ---------------------------------------------------------
    TRAIN_PATH = os.path.join(project_root, 'data', 'OCT', 'train')
    TEST_PATH = os.path.join(project_root, 'data', 'OCT', 'test')

    IMG_SIZE = 224
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
    else:
        DEVICE = torch.device("cpu")
        gpu_name = "CPU"

    print(f"Using device: {DEVICE} ({gpu_name})")

    print("Loading datasets...")
    train_loader, test_loader, normal_idx = get_anomaly_dataloaders(
        train_path=TRAIN_PATH, 
        test_path=TEST_PATH, 
        img_size=IMG_SIZE,
        n_train_normal = 1000,
        n_test_normal = 200,
        n_test_anomaly=200,
        batch_size=BATCH_SIZE
    )

    # ---------------------------------------------------------
    # 2. Models: ViT (Frozen) & Autoencoder (Trainable)
    # ---------------------------------------------------------
    print(f"Loading ViT-B/16 onto {DEVICE}...")
    print(f"Loading ViT-B/16 onto {DEVICE} ({gpu_name})...")
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity() 
    vit = vit.to(DEVICE)
    vit.eval() # Freeze ViT

    # Define the Autoencoder for the 768-dim ViT features
    class FeatureAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            self.decoder = nn.Sequential(
                nn.Linear(128, 256), 
                nn.ReLU(),
                nn.Linear(256, 768)
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    ae_model = FeatureAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

    # ---------------------------------------------------------
    # 3. Helper to Extract Features Once (Massive Speedup)
    # ---------------------------------------------------------
    def pre_extract_features(loader):
        features = []
        labels = []
        with torch.no_grad():
            for imgs, lbls in loader:
                feats = vit(imgs.to(DEVICE))
                features.append(feats.cpu())
                labels.extend(lbls.numpy())
        return torch.cat(features), np.array(labels)

    print("\nPre-extracting ViT features (happens only once)...")
    train_feats_tensor, _ = pre_extract_features(train_loader)
    test_feats_tensor, test_labels_raw = pre_extract_features(test_loader)

    # Create a fast DataLoader just for the Autoencoder training
    ae_train_loader = DataLoader(TensorDataset(train_feats_tensor), batch_size=BATCH_SIZE, shuffle=True)

    # ---------------------------------------------------------
    # 4. MLflow Tracking & Training Loop
    # ---------------------------------------------------------
    tracker = MLFlowTracker(experiment_name="ViT_Autoencoder_Anomaly")
    EPOCHS = 10

    with tracker as run:
        tracker.log_params({
            "backbone": "vit_b_16_pretrained",
            "ae_bottleneck": 64,
            "epochs": EPOCHS,
            "threshold_method": "F1-Optimization"
        })

        print("\nTraining Autoencoder...")
        train_losses = []
        for epoch in range(EPOCHS):
            ae_model.train()
            batch_losses = []
            for batch in ae_train_loader:
                feats = batch[0].to(DEVICE)
                
                recon = ae_model(feats)
                loss = criterion(recon, feats)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
                
            avg_loss = np.mean(batch_losses)
            train_losses.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

        # ---------------------------------------------------------
        # 5. Validation Split & Dynamic Threshold Optimization
        # ---------------------------------------------------------
        ae_model.eval()
        print("\nEvaluating and finding optimal threshold...")

        def get_reconstruction_errors(feature_tensor):
            with torch.no_grad():
                feats = feature_tensor.to(DEVICE)
                recon = ae_model(feats)
                errors = torch.mean((feats - recon)**2, dim=1).cpu().numpy()
            return errors

        train_errors = get_reconstruction_errors(train_feats_tensor)
        all_test_errors = get_reconstruction_errors(test_feats_tensor)

        # Ground truth: 0 for Normal, 1 for Anomaly
        y_true_all = np.array([0 if l == normal_idx else 1 for l in test_labels_raw])

        # SPLIT the test data into Validation (for finding the line) and Test (for the final grade)
        val_errs, test_errs, val_lbls, test_lbls = train_test_split(
            all_test_errors, y_true_all, test_size=0.5, stratify=y_true_all, random_state=42
        )
        # ---------------------------------------------------------
        # THRESHOLD-INDEPENDENT METRICS (on validation set)
        # ---------------------------------------------------------

        # AUROC (how well model separates normal vs anomaly)
        val_auroc = roc_auc_score(val_lbls, val_errs)

        # AUPRC (important for imbalance)
        val_auprc = average_precision_score(val_lbls, val_errs)

        tracker.log_metrics({
            "val_auroc": val_auroc,
            "val_auprc": val_auprc
        })

        print("\n--- Threshold-Independent Metrics (Validation) ---")
        print(f"AUROC: {val_auroc:.4f}")
        print(f"AUPRC: {val_auprc:.4f}")

        # print("Running F1-Maximization on Validation Set...")
        # best_thresh = 0
        # best_f1 = 0

        # # Test 1,000 different possible lines between the lowest and highest validation error
        # thresholds_to_test = np.linspace(val_errs.min(), val_errs.max(), 1000)

        # for t in thresholds_to_test:
        #     temp_preds = (val_errs > t).astype(int)
        #     _, _, f1, _ = precision_recall_fscore_support(val_lbls, temp_preds, average='binary', zero_division=0)
            
        #     if f1 > best_f1:
        #         best_f1 = f1
        #         best_thresh = t

        # tracker.log_metric("optimal_threshold", best_thresh)
        # print(f"Optimal Threshold Found: {best_thresh:.6f} (Validation F1: {best_f1:.4f})")

        print("Running Threshold Optimization with Precision Constraint...")

        best_thresh = 0
        best_f1 = 0
        best_precision = 0
        best_recall = 0

        # Handle edge case where all errors are identical
        if val_errs.min() == val_errs.max():
            best_thresh = val_errs.min()
        else:
            thresholds_to_test = np.linspace(val_errs.min(), val_errs.max(), 1000)

            MIN_PRECISION = 0.90  # you can tune this (0.85–0.95)

            for t in thresholds_to_test:
                temp_preds = (val_errs > t).astype(int)

                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_lbls, temp_preds, average='binary', zero_division=0
                )

                # Only consider thresholds that satisfy precision constraint
                if precision >= MIN_PRECISION:
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = t
                        best_precision = precision
                        best_recall = recall

            # Fallback if no threshold satisfies precision condition
            if best_f1 == 0:
                print("No threshold met precision constraint, falling back to best F1")

                for t in thresholds_to_test:
                    temp_preds = (val_errs > t).astype(int)

                    precision, recall, f1, _ = precision_recall_fscore_support(
                        val_lbls, temp_preds, average='binary', zero_division=0
                    )

                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = t
                        best_precision = precision
                        best_recall = recall

        tracker.log_metric("optimal_threshold", best_thresh)

        print(f"\nOptimal Threshold Found: {best_thresh:.6f}")
        print(f"Validation Precision: {best_precision:.4f}")
        print(f"Validation Recall: {best_recall:.4f}")
        print(f"Validation F1: {best_f1:.4f}")
        # ---------------------------------------------------------
        # 6. Final Evaluation on Unseen Test Set
        # ---------------------------------------------------------
        # Apply the perfectly calculated threshold to the actual test set
        test_preds = [1 if e > best_thresh else 0 for e in test_errs]

        precision, recall, final_f1, _ = precision_recall_fscore_support(test_lbls, test_preds, average='binary')
        tracker.log_metrics({"test_precision": precision, "test_recall": recall, "test_f1": final_f1})

        print("\n" + "="*30)
        print("FINAL PERFORMANCE (UNSEEN TEST SET)")
        target_names = ['Healthy (Normal)', 'Pathology (Anomaly)']
        cm = confusion_matrix(test_lbls, test_preds)
        print(classification_report(test_lbls, test_preds, target_names=target_names))

        # ---------------------------------------------------------
        # 7. Visualization & Artifacts
        # ---------------------------------------------------------
        MODEL_NAME = "vit_autoencoder"
        
        loss_path = plot_loss(train_losses, "ae_loss.png", MODEL_NAME)
        tracker.log_artifact(loss_path)
        
        cm_path = plot_confusion_matrix_custom(cm, target_names, "vit_ae_cm.png", MODEL_NAME)
        tracker.log_artifact(cm_path)
        
        # Filter errors for plotting based on the test labels
        test_normal_errors = test_errs[np.array(test_lbls) == 0]
        test_anomaly_errors = test_errs[np.array(test_lbls) == 1]
        
        dist_path = plot_error_distribution(
            train_errors, test_normal_errors, test_anomaly_errors, 
            best_thresh, "vit_ae_dist.png", MODEL_NAME
        )
        tracker.log_artifact(dist_path)
        
        print(f"\nResults logged to MLflow under experiment: {tracker.experiment_name}")
        plt.show()

if __name__ == "__main__":
    main()