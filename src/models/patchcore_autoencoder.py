"""
Module: patchcore_autoencoder.py
Description: Implements an anomaly detection pipeline using a ResNet18 backbone 
to extract PatchCore features, followed by a purely convolutional autoencoder 
trained with Structural Similarity Index (SSIM) loss.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
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
    generate_anomaly_analysis,
)
from src.helper.mlflow_helper import MLFlowTracker

MODEL_NAME = "patchcore_ae"
EPOCHS = 30
RUN_PARAMS = {
    "backbone": "resnet18_layer2_layer3",
    "encoder_params": "[384, 128, 64]",
    "decoder_params": "[64, 128, 384]",
    "epochs": EPOCHS,
    "learning_rate": 0.001,
    "loss_type": "Pure_SSIM",
    "batch_size": config.BATCH_SIZE, 
    "image_size": config.IMG_SIZE
}

# ---------------------------------------------------------
# 1. Feature Extraction Setup
# ---------------------------------------------------------
extracted_features = {}

def create_feature_hook(name):
    """
    Creates a forward hook to extract features from a specific model layer.
    
    Args:
        name (str): The name/key to assign to the extracted features.
        
    Returns:
        Callable: The hook function to be registered with the layer.
    """
    def hook(model, input, output):
        extracted_features[name] = output.detach()
    return hook

class FeatureAutoencoder(nn.Module):
    """
    A purely convolutional autoencoder designed to reconstruct feature maps.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1) 
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 384, kernel_size=1)
        )
        
    def forward(self, x):
        """
        Forward pass of the feature autoencoder.
        """
        return self.decoder(self.encoder(x))

def extract_features(loader, resnet_model, desc):
    """
    Passes data through the ResNet backbone to extract and pool feature maps statically.
    """
    all_patches, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images_3c = images.repeat(1, 3, 1, 1).to(config.DEVICE, non_blocking=True)
            _ = resnet_model(images_3c)
            
            f2, f3 = extracted_features['layer2'], extracted_features['layer3']
            f2 = F.interpolate(f2, size=f3.shape[2:], mode='bilinear', align_corners=False)
            f_cat = torch.cat([f2, f3], dim=1) 
            f_pool = F.avg_pool2d(f_cat, kernel_size=3, stride=1, padding=1)
            
            all_patches.append(f_pool.cpu())
            all_labels.extend(labels.numpy())
            
    return torch.cat(all_patches), np.array(all_labels)


if __name__ == "__main__":
    print(f"Starting SSIM Pipeline on {config.DEVICE}...")

    train_loader, test_loader, normal_idx = dataloader(
        train_path=config.TRAIN_PATH, 
        test_path=config.TEST_PATH, 
        img_size=config.IMG_SIZE,
        n_train_normal= config.N_TRAIN_NORMAL, 
        n_test_normal= config.N_TEST_NORMAL,
        n_test_anomaly_per_class=config.N_TEST_ANOMALY_PER_CLASS, 
        batch_size=config.BATCH_SIZE
    )
    # ---------------------------------------------------------
    # 2. Models: ResNet Backbone & Autoencoder
    # ---------------------------------------------------------
    print("Loading ResNet18 Backbone...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(config.DEVICE)
    resnet.eval()
    
    # Register the hooks
    resnet.layer2.register_forward_hook(create_feature_hook('layer2')) 
    resnet.layer3.register_forward_hook(create_feature_hook('layer3')) 

    ae_model = FeatureAutoencoder().to(config.DEVICE)
    #ae_model = torch.compile(ae_model) 

    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda') 

    # ---------------------------------------------------------
    # 3. Static Feature Extraction
    # ---------------------------------------------------------
    print("\nExtracting PatchCore features to RAM...")
    train_features_tensor, _ = extract_features(train_loader, resnet, "Extracting Train")
    test_features_tensor, test_labels_raw = extract_features(test_loader, resnet, "Extracting Test")

    ae_train_loader = DataLoader(TensorDataset(train_features_tensor), batch_size=config.BATCH_SIZE, shuffle=True)

    # ---------------------------------------------------------
    # 4. Training Loop (Pure SSIM Loss)
    # ---------------------------------------------------------
    tracker = MLFlowTracker(experiment_name="PatchCore_Autoencoder_SSIM")

    with tracker as run:
        tracker.log_params(RUN_PARAMS)
        print("\nTraining Autoencoder with Pure SSIM Loss...")
        train_losses = []
        
        for epoch in range(EPOCHS):
            ae_model.train()
            for batch in ae_train_loader:
                features = batch[0].to(config.DEVICE, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    recon = ae_model(features)
                    
                    # Reshape for SSIM: PatchCore is [B, 384, 28, 28]
                    recon_2d = torch.clamp(recon, 0, 1)
                    features_2d = torch.clamp(features, 0, 1)
                    
                    # Calculate 1 - SSIM
                    ssim_loss = 1 - ssim(recon_2d, features_2d, data_range=1.0)
                
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
        
        def calculate_ssim_errors(feature_tensor):
            """Calculates SSIM-based reconstruction errors for a given feature tensor."""
            with torch.no_grad():
                features = feature_tensor.to(config.DEVICE)
                recon = ae_model(features)
                
                r2d = torch.clamp(recon, 0, 1)
                f2d = torch.clamp(features, 0, 1)
                
                sample_errors = []
                for i in range(r2d.shape[0]):
                    val = ssim(r2d[i:i+1], f2d[i:i+1], data_range=1.0)
                    sample_errors.append((1 - val).item())
                return np.array(sample_errors)

        print("\nCalculating SSIM Errors for evaluation...")
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
        # 6. Logging & Artifacts (Standardized Metrics)
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
        tracker.log_metrics({"optimal_threshold": float(best_thresh)})
        tracker.log_metrics({
            "precision": precision, 
            "recall": recall, 
            "f1": f1,
            "specificity": specificity,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr
        })
        
        torch.save(ae_model.state_dict(), os.path.join(config.PROJECT_ROOT, "patchcore_ae_model.pth"))
        
        # ---------------------------------------------------------
        # 7. Analysis: Heatmaps of Top 10 & Worst 10
        # ---------------------------------------------------------
        print("\nGenerating Heatmap Comparisons...")

        raw_images_tensor = next(iter(DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset))))[0]
        generate_anomaly_analysis(
            model=ae_model,
            feature_tensor=test_features_tensor,
            raw_images_tensor=raw_images_tensor,
            all_errors=all_test_errors,
            y_true=y_true_all,
            model_name=MODEL_NAME,
            view_mode="overlay",
        )

        print("\n" + "="*30)
        print(classification_report(test_labels, test_preds, target_names=['Normal', 'Anomaly']))
