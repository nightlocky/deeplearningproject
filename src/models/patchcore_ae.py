import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
import cv2

# Import Config and Helpers
import config
from dataLoader.dataLoader import dataloader
from helper.visualization_helper import plot_loss, plot_error_distribution, plot_confusion_matrix, plot_anomaly_comparison
from helper.mlflow_helper import MLFlowTracker

MODEL_NAME = "patchcore_ae"
EPOCHS = 20
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
# 1. Data Loading (Using Universal Config)
# ---------------------------------------------------------
print(f"Starting SSIM Pipeline on {config.DEVICE}...")

train_loader, test_loader, normal_idx = dataloader(
    train_path=config.TRAIN_PATH, 
    test_path=config.TEST_PATH, 
    img_size=config.IMG_SIZE,
    n_train_normal=config.N_TRAIN_NORMAL,
    n_test_normal=config.N_TEST_NORMAL,
    n_test_anomaly=config.N_TEST_ANOMALY,
    batch_size=config.BATCH_SIZE
)

# ---------------------------------------------------------
# 2. Models: ResNet Backbone & Autoencoder
# ---------------------------------------------------------
print("Loading ResNet18 Backbone...")
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(config.DEVICE)
resnet.eval()

features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

resnet.layer2.register_forward_hook(get_features('layer2')) 
resnet.layer3.register_forward_hook(get_features('layer3')) 

class FeatureAutoencoder(nn.Module):
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
        return self.decoder(self.encoder(x))

ae_model = FeatureAutoencoder().to(config.DEVICE)
ae_model = torch.compile(ae_model) 

optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler('cuda') 

# ---------------------------------------------------------
# 3. Static Feature Extraction
# ---------------------------------------------------------
def pre_extract_features(loader, desc):
    all_patches, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=desc):
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            _ = resnet(imgs)
            f2, f3 = features['layer2'], features['layer3']
            f2 = F.interpolate(f2, size=f3.shape[2:], mode='bilinear', align_corners=False)
            f_cat = torch.cat([f2, f3], dim=1) 
            f_pool = F.avg_pool2d(f_cat, kernel_size=3, stride=1, padding=1)
            
            all_patches.append(f_pool.cpu())
            all_labels.extend(lbls.numpy())
    return torch.cat(all_patches), np.array(all_labels)

print("\nExtracting PatchCore features to RAM...")
train_feats_tensor, _ = pre_extract_features(train_loader, "Extracting Train")
test_feats_tensor, test_labels_raw = pre_extract_features(test_loader, "Extracting Test")

ae_train_loader = DataLoader(TensorDataset(train_feats_tensor), batch_size=config.BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------
# 4. Training Loop (Pure SSIM Loss)
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="PatchCore_Autoencoder")

with tracker as run:
    tracker.log_params(RUN_PARAMS)
    print("\nTraining Autoencoder with Pure SSIM Loss...")
    train_losses = []
    
    for epoch in range(EPOCHS):
        ae_model.train()
        batch_losses = []
        for batch in ae_train_loader:
            feats = batch[0].to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                recon = ae_model(feats)
                
                # Reshape for SSIM: PatchCore is [B, 384, 28, 28]
                recon_2d = torch.clamp(recon, 0, 1)
                feats_2d = torch.clamp(feats, 0, 1)
                
                # Calculate 1 - SSIM
                ssim_loss = 1 - ssim(recon_2d, feats_2d, data_range=1.0)
            
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
            feats = feature_tensor.to(config.DEVICE)
            recon = ae_model(feats)
            
            r2d = torch.clamp(recon, 0, 1)
            f2d = torch.clamp(feats, 0, 1)
            
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
        all_test_errors, y_true_all, test_size=0.5, stratify=y_true_all, random_state=CONFIG.RANDOM_SEED
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
    l_p = plot_loss(train_losses, "ae_loss.png", MODEL_NAME)

    cm_p = plot_confusion_matrix(
        cm=confusion_matrix(test_lbls, test_preds), 
        target_names=['Normal', 'Anomaly'], 
        precision=precision, 
        recall=recall, 
        f1=f1,
        save_path="ae_cm.png", 
        model_name=MODEL_NAME
    )
    
    d_p = plot_error_distribution(
        train_errors, 
        test_errs[test_lbls==0], 
        test_errs[test_lbls==1], 
        best_thresh, 
        "ae_dist.png", 
        MODEL_NAME
    )

    tracker.log_artifact(l_p)
    tracker.log_artifact(cm_p)
    tracker.log_artifact(d_p)
    tracker.log_metric("optimal_threshold", best_thresh)
    tracker.log_metrics({"precision": precision, "recall": recall, "f1": f1})
    
    torch.save(ae_model.state_dict(), os.path.join(config.PROJECT_ROOT, "patchcore_ae_model.pth"))
    
    # ---------------------------------------------------------
    # 7. Deep Analysis: Heatmaps of Top 10 & Worst 10
    # ---------------------------------------------------------
    print("\nGenerating Heatmap Comparisons...")
    
    raw_images_tensor = next(iter(DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset))))[0]
    
    with torch.no_grad():
        # Get reconstructions for the full test set
        recon_features = ae_model(test_feats_tensor.to(config.DEVICE))
        
        # Calculate spatial squared error [B, 1, 28, 28]
        spatial_error = torch.mean((test_feats_tensor.to(config.DEVICE) - recon_features)**2, dim=1, keepdim=True)
        
        # Interpolate error heatmap up to 224x224 to match original pixels
        upscaled_error = F.interpolate(spatial_error, size=(config.IMG_SIZE, config.IMG_SIZE), mode='bilinear')
        test_recons = upscaled_error.cpu().numpy()
    
    raw_images = raw_images_tensor.numpy()
    
    anomaly_mask = (y_true_all == 1)
    anom_scores = all_test_errors[anomaly_mask]
    real_indices = np.where(anomaly_mask)[0]
    
    top_hits_sub_idx = np.argsort(anom_scores)[-10:][::-1]
    top_hits_indices = real_indices[top_hits_sub_idx]
    
    top_miss_sub_idx = np.argsort(anom_scores)[:10]
    top_miss_indices = real_indices[top_miss_sub_idx]
    
    plot_anomaly_comparison(
        raw_images, test_recons, all_test_errors, top_hits_indices,
        "Top 10 Correctly Identified Anomalies", "heatmaps_best_hits.png", MODEL_NAME
    )
    
    plot_anomaly_comparison(
        raw_images, test_recons, all_test_errors, top_miss_indices,
        "Top 10 Missed Anomalies (False Negatives)", "heatmaps_worst_misses.png", MODEL_NAME
    )

    print("\n" + "="*30)
    print(classification_report(test_lbls, test_preds, target_names=['Normal', 'Anomaly']))