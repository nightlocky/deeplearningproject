import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
# Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir) # Targets /workspace
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
BATCH_SIZE = 64 
NUM_WORKERS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starting SSIM Pipeline on {DEVICE}...")
train_loader, test_loader, normal_idx = get_anomaly_dataloaders(
    train_path=TRAIN_PATH, 
    test_path=TEST_PATH, 
    img_size=IMG_SIZE,
    n_train_normal=50000,
    n_test_normal=5000,
    n_test_anomaly=750,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)
# ---------------------------------------------------------
# 2. Models: ViT Backbone & Autoencoder
# ---------------------------------------------------------
print("Loading ViT-B/16 Backbone...")
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
vit.heads = nn.Identity() 
vit = vit.to(DEVICE)
vit.eval() 
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
ae_model = torch.compile(ae_model) 
optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler('cuda') 
# ---------------------------------------------------------
# 3. Static Feature Extraction
# ---------------------------------------------------------
def pre_extract_features(loader, desc):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=desc):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feats = vit(imgs.to(DEVICE, non_blocking=True))
            features.append(feats.float().cpu())
            labels.extend(lbls.numpy())
    return torch.cat(features), np.array(labels)
print("\nExtracting ViT features to RAM...")
train_feats_tensor, _ = pre_extract_features(train_loader, "Extracting Train")
test_feats_tensor, test_labels_raw = pre_extract_features(test_loader, "Extracting Test")
ae_train_loader = DataLoader(TensorDataset(train_feats_tensor), batch_size=BATCH_SIZE, shuffle=True)
# ---------------------------------------------------------
# 4. Training Loop (Pure SSIM Loss)
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="ViT_Autoencoder_SSIM")
EPOCHS = 20
with tracker as run:
    tracker.log_params({
        "backbone": "vit_b_16",
        "ae_bottleneck": 128,
        "epochs": EPOCHS,
        "loss_type": "Pure_SSIM"
    })
    print("\nTraining Autoencoder with Pure SSIM Loss...")
    train_losses = []
    for epoch in range(EPOCHS):
        ae_model.train()
        batch_losses = []
        for batch in ae_train_loader:
            feats = batch[0].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                recon = ae_model(feats)
                
                # Reshape (B, 768) -> (B, 1, 24, 32) for SSIM
                recon_2d = torch.clamp(recon.view(-1, 1, 24, 32), 0, 1)
                feats_2d = torch.clamp(feats.view(-1, 1, 24, 32), 0, 1)
                
                # Calculate 1 - SSIM
                ssim_val = ssim(recon_2d, feats_2d, data_range=1.0)
                ssim_loss = 1 - ssim_val
            
            scaler.scale(ssim_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(ssim_loss.item())
            
        avg_loss = np.mean(batch_losses)
        train_losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss (SSIM): {avg_loss:.6f}")
    # ---------------------------------------------------------
    # 5. Threshold Optimization (Using SSIM Error)
    # ---------------------------------------------------------
    ae_model.eval()
    
    def get_ssim_errors(feature_tensor):
        with torch.no_grad():
            feats = feature_tensor.to(DEVICE)
            recon = ae_model(feats)
            
            r2d = torch.clamp(recon.view(-1, 1, 24, 32), 0, 1)
            f2d = torch.clamp(feats.view(-1, 1, 24, 32), 0, 1)
            
            # Calculate error per image
            sample_errors = []
            for i in range(r2d.shape[0]):
                val = ssim(r2d[i:i+1], f2d[i:i+1], data_range=1.0)
                sample_errors.append((1 - val).item())
            return np.array(sample_errors)
    print("\nCalculating SSIM Errors for evaluation...")
    train_errors = get_ssim_errors(train_feats_tensor)
    all_test_errors = get_ssim_errors(test_feats_tensor)
    y_true_all = np.array([0 if l == normal_idx else 1 for l in test_labels_raw])
    val_errs, test_errs, val_lbls, test_lbls = train_test_split(
        all_test_errors, y_true_all, test_size=0.5, stratify=y_true_all, random_state=42
    )
    thresholds = np.linspace(val_errs.min(), val_errs.max(), 1000)
    best_f1, best_thresh = 0, 0
    for t in thresholds:
        preds = (val_errs > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(val_lbls, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    test_preds = [1 if e > best_thresh else 0 for e in test_errs]
    precision, recall, f1, _ = precision_recall_fscore_support(test_lbls, test_preds, average='binary')
    
    # ---------------------------------------------------------
    # 6. Logging & Artifacts
    # ---------------------------------------------------------
    os.chdir(project_root) 
    MODEL_NAME = "vit_autoencoder"
    
    l_p = plot_loss(train_losses, "ae_loss.png", MODEL_NAME)
    cm_p = plot_confusion_matrix_custom(confusion_matrix(test_lbls, test_preds), ['Normal', 'Anomaly'], "ae_cm.png", MODEL_NAME)
    d_p = plot_error_distribution(train_errors, test_errs[test_lbls==0], test_errs[test_lbls==1], best_thresh, "ae_dist.png", MODEL_NAME)
    tracker.log_artifact(l_p)
    tracker.log_artifact(cm_p)
    tracker.log_artifact(d_p)
    tracker.log_metric("optimal_threshold", best_thresh)
    tracker.log_metrics({"precision": precision, "recall": recall, "f1": f1})
    
    torch.save(ae_model.state_dict(), os.path.join(project_root, "vit_ae_model.pth"))
    
    print("\n" + "="*30)
    print("FINAL PERFORMANCE (PURE SSIM)")
    print(classification_report(test_lbls, test_preds, target_names=['Normal', 'Anomaly']))