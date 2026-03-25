import os
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
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
# 2. Models: ViT (Frozen) & Autoencoder (Trainable)
# ---------------------------------------------------------
print(f"Loading ViT-B/16 onto {DEVICE}...")
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
            nn.Linear(256, 64) # Bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
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
EPOCHS = 30
PERCENTILE_THRESHOLD = 90 # Lowered to 90 to improve Recall!

with tracker as run:
    tracker.log_params({
        "backbone": "vit_b_16_pretrained",
        "ae_bottleneck": 64,
        "epochs": EPOCHS,
        "threshold_percentile": PERCENTILE_THRESHOLD
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

    # --- Step 5: Evaluation & Thresholding ---
    ae_model.eval()
    print("\nEvaluating on Test Set...")
    
    def get_reconstruction_errors(feature_tensor):
        with torch.no_grad():
            feats = feature_tensor.to(DEVICE)
            recon = ae_model(feats)
            # Calculate MSE per image (mean across the 768 dimensions)
            errors = torch.mean((feats - recon)**2, dim=1).cpu().numpy()
        return errors

    train_errors = get_reconstruction_errors(train_feats_tensor)
    test_errors = get_reconstruction_errors(test_feats_tensor)

    # Set threshold based on normal training data distribution
    THRESHOLD = np.percentile(train_errors, PERCENTILE_THRESHOLD)
    tracker.log_metric("error_threshold", THRESHOLD)

    # Ground truth: 0 for Normal, 1 for Anomaly
    y_true = [0 if l == normal_idx else 1 for l in test_labels_raw]
    # Prediction: 1 if error > threshold
    y_pred = [1 if e > THRESHOLD else 0 for e in test_errors]

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    tracker.log_metrics({"precision": precision, "recall": recall, "f1_score": f1})

    print("\n" + "="*30)
    print("FINAL PERFORMANCE (ViT + AUTOENCODER)")
    target_names = ['Healthy (Normal)', 'Pathology (Anomaly)']
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ---------------------------------------------------------
    # 6. Visualization & Artifacts
    # ---------------------------------------------------------
    MODEL_NAME = "vit_autoencoder"
    
    loss_path = plot_loss(train_losses, "ae_loss.png", MODEL_NAME)
    tracker.log_artifact(loss_path)
    
    cm_path = plot_confusion_matrix_custom(cm, target_names, "vit_ae_cm.png", MODEL_NAME)
    tracker.log_artifact(cm_path)
    
    test_normal_errors = test_errors[np.array(y_true) == 0]
    test_anomaly_errors = test_errors[np.array(y_true) == 1]
    
    dist_path = plot_error_distribution(
        train_errors, test_normal_errors, test_anomaly_errors, 
        THRESHOLD, "vit_ae_dist.png", MODEL_NAME
    )
    tracker.log_artifact(dist_path)
    
    print(f"\nResults logged to MLflow under experiment: {tracker.experiment_name}")
    plt.show()