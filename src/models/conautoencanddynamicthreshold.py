import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, using simple loop")
    def tqdm(iterable, **kwargs):
        return iterable

# Ensure the parent directory (src) and its data subdirectory are in the path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)

# Import custom helpers
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
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading datasets...")
train_loader, raw_test_loader, normal_idx = get_anomaly_dataloaders(
    train_path=TRAIN_PATH, 
    test_path=TEST_PATH, 
    img_size=IMG_SIZE,
    n_train_normal=10000, 
    n_test_normal=250,    
    n_test_anomaly=750,   
    batch_size=BATCH_SIZE
)

# ---------------------------------------------------------
# 2. Model: Convolutional Autoencoder
# ---------------------------------------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = ConvAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------
# 3. MLflow Tracking & Training Loop
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="CAE_From_Scratch")

with tracker as run:
    tracker.log_params({"model": "ConvAutoencoder", "epochs": EPOCHS, "lr": 1e-3})

    print(f"\nTraining ConvAutoencoder on {DEVICE}...")
    train_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        
        # Wrapped the inner loop with tqdm
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
        
        for imgs, _ in loop:
            imgs = imgs.to(DEVICE)
            
            recon = model(imgs)
            loss = criterion(recon, imgs) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            
            # Update the progress bar text with the live loss
            loop.set_postfix(loss=loss.item())
            
        avg_loss = np.mean(batch_losses)
        train_losses.append(avg_loss)
        tracker.log_metric("train_loss", avg_loss, step=epoch)
        # Print final epoch summary to the console so it stays after the progress bar clears
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {avg_loss:.6f}")

    # --- Save Model ---
    model_path = "cae_model.pth"
    torch.save(model.state_dict(), model_path)
    tracker.log_artifact(model_path)
    print(f"\nModel weights saved to {model_path}")

    # ---------------------------------------------------------
    # 4. Feature Extraction & Validation Split
    # ---------------------------------------------------------
    model.eval()
    print("\nCalculating Reconstruction Errors...")
    
    def get_errors_and_labels(loader, desc="Processing"):
        errors, labels = [], []
        with torch.no_grad():
            # Wrapped this loop with tqdm too!
            for imgs, lbls in tqdm(loader, desc=desc):
                imgs = imgs.to(DEVICE)
                recon = model(imgs)
                batch_err = torch.mean((imgs - recon)**2, dim=[1,2,3]).cpu().numpy()
                errors.extend(batch_err)
                labels.extend(lbls.numpy())
        return np.array(errors), np.array(labels)

    # Note: added description tags to the tqdm calls so you know which dataset it is processing
    train_errors, _ = get_errors_and_labels(train_loader, desc="Train Set Errors")
    all_test_errors, all_test_labels = get_errors_and_labels(raw_test_loader, desc="Test Set Errors")
    
    y_true_all = np.array([0 if l == normal_idx else 1 for l in all_test_labels])

    val_errs, test_errs, val_lbls, test_lbls = train_test_split(
        all_test_errors, y_true_all, test_size=0.5, stratify=y_true_all, random_state=42
    )

    # ---------------------------------------------------------
    # 5. Dynamic Threshold Optimization
    # ---------------------------------------------------------
    print("\nOptimizing threshold on Validation Set...")
    best_thresh = 0
    best_f1 = 0
    thresholds_to_test = np.linspace(val_errs.min(), val_errs.max(), 1000)
    
    for t in thresholds_to_test:
        temp_preds = (val_errs > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(val_lbls, temp_preds, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    tracker.log_metric("optimal_threshold", best_thresh)
    print(f"Optimal Threshold Found: {best_thresh:.6f} (Val F1: {best_f1:.4f})")

    # ---------------------------------------------------------
    # 6. Final Evaluation on Unseen Test Set
    # ---------------------------------------------------------
    test_preds = (test_errs > best_thresh).astype(int)

    precision, recall, final_f1, _ = precision_recall_fscore_support(test_lbls, test_preds, average='binary')
    tracker.log_metrics({"test_precision": precision, "test_recall": recall, "test_f1": final_f1})

    print("\n" + "="*30)
    print("FINAL PERFORMANCE ON UNSEEN TEST SET")
    target_names = ['Healthy (Normal)', 'Pathology (Anomaly)']
    cm = confusion_matrix(test_lbls, test_preds)
    print(classification_report(test_lbls, test_preds, target_names=target_names))

    # ---------------------------------------------------------
    # 7. Visualization & Artifacts
    # ---------------------------------------------------------
    MODEL_NAME = "cae_from_scratch"
    
    loss_path = plot_loss(train_losses, "cae_loss.png", MODEL_NAME)
    tracker.log_artifact(loss_path)
    
    cm_path = plot_confusion_matrix_custom(cm, target_names, "cae_cm.png", MODEL_NAME)
    tracker.log_artifact(cm_path)
    
    dist_path = plot_error_distribution(
        train_errors, 
        test_errs[test_lbls == 0], 
        test_errs[test_lbls == 1], 
        best_thresh, "cae_dist.png", MODEL_NAME
    )
    tracker.log_artifact(dist_path)
    
    print(f"\nResults logged to MLflow under experiment: {tracker.experiment_name}")
    plt.show()