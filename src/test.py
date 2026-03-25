import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Import custom helpers
from visualization_helper import (
    plot_loss, 
    plot_reconstruction_comparison, 
    plot_error_distribution, 
    plot_confusion_matrix_custom
)
from mlflow_helper import MLFlowTracker

# ---------------------------------------------------------
# 1. Configuration & Data Loading
# ---------------------------------------------------------
TRAIN_PATH = r'../data/OCT/train'
TEST_PATH = r'../data/OCT/test'
IMG_SIZE = 128 
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 25

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Load datasets
full_train_ds = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
full_test_ds = datasets.ImageFolder(root=TEST_PATH, transform=transform)

# NORMAL class index
normal_idx = full_train_ds.class_to_idx['NORMAL']

# Train Set: 500 NORMAL images
train_normal_indices = [i for i, (_, label) in enumerate(full_train_ds.samples) if label == normal_idx]
train_subset = Subset(full_train_ds, train_normal_indices[:500]) 
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

# Test Set: 250 NORMAL + 30 Anomalies
test_normal_indices = [i for i, (_, label) in enumerate(full_test_ds.samples) if label == normal_idx]
test_anomaly_indices = [i for i, (_, label) in enumerate(full_test_ds.samples) if label != normal_idx]

test_subset_indices = test_normal_indices[:250] + test_anomaly_indices[:30]
test_ds = Subset(full_test_ds, test_subset_indices)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print(f"Dataset Ready!")
print(f"-> Training on {len(train_subset)} Normal images.")
print(f"-> Testing on {len(test_ds)} images (Imbalanced: 250 Normal vs 30 Abnormal).")

# ---------------------------------------------------------
# 2. Architecture
# ---------------------------------------------------------
class OCTAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x): 
        return self.decoder(self.encoder(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OCTAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------------------------------------
# 3. Training & MLflow Tracking
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="OCT_Anomaly_Detection_Separated")

with tracker as run:
    # Log hyperparameters
    tracker.log_params({
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "device": str(device)
    })
    
    print(f"\nStarting Training ({EPOCHS} Epochs)...")
    model.train()
    epoch_losses = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        tracker.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    # ---------------------------------------------------------
    # 4. Evaluation
    # ---------------------------------------------------------
    model.eval()
    print("\nCalculating Anomaly Threshold...")

    def get_reconstruction_error(img_tensor):
        img_tensor = img_tensor.unsqueeze(0).to(device)
        recon = model(img_tensor)
        return criterion(recon, img_tensor).item()

    # Determine Threshold
    train_errors = []
    with torch.no_grad():
        for i in range(len(train_subset)):
            img, _ = train_subset[i]
            train_errors.append(get_reconstruction_error(img))

    threshold = np.mean(train_errors) + (2 * np.std(train_errors))
    tracker.log_metric("anomaly_threshold", threshold)
    print(f"Calculated Threshold: {threshold:.5f}")

    # Test
    y_true, y_pred = [], []
    test_normal_errors, test_anomaly_errors = [], []

    with torch.no_grad():
        for i in range(len(test_ds)):
            img, label = test_ds[i]
            actual_is_anomaly = 0 if label == normal_idx else 1
            y_true.append(actual_is_anomaly)
            
            error = get_reconstruction_error(img)
            if actual_is_anomaly == 0:
                test_normal_errors.append(error)
            else:
                test_anomaly_errors.append(error)
            
            y_pred.append(1 if error > threshold else 0)

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    tracker.log_metrics({
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print("\n" + "="*30)
    print("FINAL PERFORMANCE METRICS")
    cm = confusion_matrix(y_true, y_pred)
    target_names = ['Healthy (Normal)', 'Pathology (Anomaly)']
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ---------------------------------------------------------
    # 5. Visualizations & Artifacts
    # ---------------------------------------------------------
    print("\nLogging plots and model...")
    MODEL_NAME = "autoencoder"
    
    # Loss plot
    loss_path = plot_loss(epoch_losses, save_path="loss_plot.png", model_name=MODEL_NAME)
    tracker.log_artifact(loss_path)
    
    # Reconstruction plot
    with torch.no_grad():
        norm_img, _ = full_test_ds[test_normal_indices[0]]
        anom_img, _ = full_test_ds[test_anomaly_indices[0]]
        recon_n = model(norm_img.unsqueeze(0).to(device)).cpu().squeeze()
        recon_a = model(anom_img.unsqueeze(0).to(device)).cpu().squeeze()
    recon_path = plot_reconstruction_comparison(norm_img, recon_n, anom_img, recon_a, save_path="reconstruction.png", model_name=MODEL_NAME)
    tracker.log_artifact(recon_path)
    
    # Error distribution
    dist_path = plot_error_distribution(train_errors, test_normal_errors, test_anomaly_errors, threshold, save_path="error_dist.png", model_name=MODEL_NAME)
    tracker.log_artifact(dist_path)
    
    # Confusion matrix
    cm_path = plot_confusion_matrix_custom(cm, target_names, save_path="confusion_matrix.png", model_name=MODEL_NAME)
    tracker.log_artifact(cm_path)
    
    # Log model
    tracker.log_model(model)
    
    plt.show()

print("\nTracking complete. Run 'mlflow ui' to see results.")
