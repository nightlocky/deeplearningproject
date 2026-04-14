"""
Module: vit_autoencoder.py
Description: Extracts embedded features from a pretrained Vision Transformer (ViT-B/16),
then trains a fully-connected autoencoder on those embeddings using SSIM.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src import config
from src.dataLoader.data_loader import dataloader
from src.helper.visualization_helper import (
    plot_loss, 
    plot_error_distribution, 
    plot_confusion_matrix, 
    generate_anomaly_analysis
)
from src.helper.mlflow_helper import MLFlowTracker

# ---------------------------------------------------------
# 1. Configuration & Autoencoder Definition
# ---------------------------------------------------------
MODEL_NAME = "vit_autoencoder"
EPOCHS = 20
RUN_PARAMS = {
    "backbone": "vit_b_16",
    "encoder_params": "[768, 256, 128]",
    "decoder_params": "[128, 256, 768]",
    "epochs": EPOCHS,
    "learning_rate": 0.001,
    "loss_type": "Pure_SSIM",
    "batch_size": config.BATCH_SIZE, 
    "image_size": config.IMG_SIZE
}

class FeatureAutoencoder(nn.Module):
    """
    A fully connected autoencoder compressing and reconstructing the 1D feature vectors 
    extracted by the ViT.
    """
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
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): 1D Feature Vector.
            
        Returns:
            torch.Tensor: Reconstructed 1D Feature Vector.
        """
        return self.decoder(self.encoder(x))

def extract_features(loader, vit_model, desc):
    """
    Extracts high-level semantic embeddings from images using the ViT.
    
    Args:
        loader (DataLoader): PyTorch DataLoader.
        vit_model (nn.Module): Pretrained Vision Transformer.
        desc (str): Progress bar descriptor.
        
    Returns:
        tuple: (torch.Tensor of features, np.ndarray of labels)
    """
    features, labels = [], []
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc=desc):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # Expand channels from 1 to 3 since dataloader is grayscale but ViT expects RGB
                images_3c = images.repeat(1, 3, 1, 1).to(config.DEVICE, non_blocking=True)
                feats = vit_model(images_3c)
            features.append(feats.float().cpu())
            labels.extend(batch_labels.numpy())
    return torch.cat(features), np.array(labels)

# =========================================================
# EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print(f"Starting ViT SSIM Pipeline on {config.DEVICE}...")

    train_loader, test_loader, normal_idx = dataloader(
        train_path=config.TRAIN_PATH, 
        test_path=config.TEST_PATH, 
        img_size=config.IMG_SIZE,
        n_train_normal=config.N_TRAIN_NORMAL,
        n_test_normal=config.N_TEST_NORMAL,
        n_test_anomaly_per_class=config.N_TEST_ANOMALY_PER_CLASS,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # ---------------------------------------------------------
    # 3. Models: ViT Backbone & Autoencoder Initialization
    # ---------------------------------------------------------
    print("Loading ViT-B/16 Backbone...")
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity() 
    vit = vit.to(config.DEVICE)
    vit.eval() 

    ae_model = FeatureAutoencoder().to(config.DEVICE)
    ae_model = torch.compile(ae_model) 

    optimizer = optim.Adam(ae_model.parameters(), lr=RUN_PARAMS["learning_rate"])
    scaler = torch.amp.GradScaler('cuda') 

    # ---------------------------------------------------------
    # 4. Static Feature Extraction
    # ---------------------------------------------------------
    print("\nExtracting ViT features to RAM...")
    train_features_tensor, _ = extract_features(train_loader, vit, "Extracting Train")
    test_features_tensor, test_labels_raw = extract_features(test_loader, vit, "Extracting Test")

    ae_train_loader = DataLoader(TensorDataset(train_features_tensor), batch_size=config.BATCH_SIZE, shuffle=True)

    # ---------------------------------------------------------
    # 5. Training Loop (Pure SSIM Loss)
    # ---------------------------------------------------------
    tracker = MLFlowTracker(experiment_name="ViT_Autoencoder")

    with tracker as run:
        tracker.log_params(RUN_PARAMS)
        print("\nTraining Autoencoder with Pure SSIM Loss...")
        train_losses = []
        
        for epoch in range(EPOCHS):
            ae_model.train()
            batch_losses = []
            for batch in ae_train_loader:
                features = batch[0].to(config.DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    recon = ae_model(features)
                    # Reshape 1D vector (B, 768) -> 2D (B, 1, 24, 32) for SSIM calculation
                    recon_2d = torch.clamp(recon.view(-1, 1, 24, 32), 0, 1)
                    features_2d = torch.clamp(features.view(-1, 1, 24, 32), 0, 1)
                    ssim_loss = 1 - ssim(recon_2d, features_2d, data_range=1.0)
                
                scaler.scale(ssim_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_losses.append(ssim_loss.item())
                
            train_losses.append(np.mean(batch_losses))
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Loss (SSIM): {train_losses[-1]:.6f}")

        # ---------------------------------------------------------
        # 6. Threshold Optimization
        # ---------------------------------------------------------
        ae_model.eval()
        
        def calculate_ssim_errors(feature_tensor, batch_size=config.BATCH_SIZE):
            """Calculates SSIM reconstruction errors between ViT embeddings."""
            sample_errors = []
            with torch.no_grad():
                feature_loader = DataLoader(TensorDataset(feature_tensor), batch_size=batch_size, shuffle=False)
                for (feature_batch,) in tqdm(feature_loader, desc="Scoring SSIM", leave=False):
                    features = feature_batch.to(config.DEVICE, non_blocking=True)
                    recon = ae_model(features)
                    
                    # Reshape back to 2D for error calculation
                    r2d = torch.clamp(recon.view(-1, 1, 24, 32), 0, 1)
                    f2d = torch.clamp(features.view(-1, 1, 24, 32), 0, 1)
                    
                    for i in range(r2d.shape[0]):
                        val = ssim(r2d[i:i+1], f2d[i:i+1], data_range=1.0)
                        sample_errors.append((1 - val).item())
            return np.array(sample_errors)

        print("\nCalculating SSIM Errors...")
        train_errors = calculate_ssim_errors(train_features_tensor)
        all_test_errors = calculate_ssim_errors(test_features_tensor)
        y_true_all = np.array([0 if l == normal_idx else 1 for l in test_labels_raw])

        val_errors, test_errors, val_labels, test_labels = train_test_split(
            all_test_errors, y_true_all, test_size=0.5, stratify=y_true_all, random_state=42
        )

        thresholds = np.linspace(val_errors.min(), val_errors.max(), 1000)
        best_f1, best_thresh = 0, 0
        for t in thresholds:
            preds = (val_errors > t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(val_labels, preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        test_preds = np.array([1 if e > best_thresh else 0 for e in test_errors])
        
        # ---------------------------------------------------------
        # 7. Logging & Standard Visuals
        # ---------------------------------------------------------
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary', zero_division=0)
        
        cm = confusion_matrix(test_labels, test_preds)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        auc_roc = roc_auc_score(test_labels, test_errors)
        auc_pr = average_precision_score(test_labels, test_errors)
        
        l_p = plot_loss(train_losses, "ae_loss.png", MODEL_NAME)
        cm_p = plot_confusion_matrix(
            cm=cm, 
            target_names=['Normal', 'Anomaly'], 
            precision=precision, 
            recall=recall, 
            f1=f1,
            specificity=specificity,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            save_path="ae_cm.png", 
            model_name=MODEL_NAME
        )
        d_p = plot_error_distribution(
            train_errors, 
            test_errors[test_labels==0], 
            test_errors[test_labels==1], 
            best_thresh, 
            "ae_dist.png", 
            MODEL_NAME
        )

        tracker.log_artifact(l_p)
        tracker.log_artifact(cm_p)
        tracker.log_artifact(d_p)
        tracker.log_metrics({
            "precision": precision, 
            "recall": recall, 
            "f1": f1,
            "specificity": specificity,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr
        })
        
        # ---------------------------------------------------------
        # 8. Deep Analysis: Heatmaps of Top 10 & Worst 10
        # ---------------------------------------------------------
        # Fetch raw images from the dataset for visualization
        raw_imgs, _ = next(iter(DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset))))
        
        generate_anomaly_analysis(
            model=ae_model,
            feature_tensor=test_features_tensor, 
            raw_images_tensor=raw_imgs,       
            all_errors=all_test_errors,       
            y_true=y_true_all,                
            model_name=MODEL_NAME,
            view_mode="overlay",
        )
        
        torch.save(ae_model.state_dict(), os.path.join(config.PROJECT_ROOT, "vit_ae_model.pth"))
        print("\n" + "="*30)
        print(classification_report(test_labels, test_preds, target_names=['Normal', 'Anomaly']))
