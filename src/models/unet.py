import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from dataLoader.dataLoader import dataloader
from helper.visualization_helper import (
    plot_loss, 
    plot_error_distribution, 
    plot_confusion_matrix
)
from helper.mlflow_helper import MLFlowTracker
# Updated import to match your new file location
from helper.EarlyStopping import EarlyStopping 

# ---------------------------------------------------------
# 0. Global Settings & Params
# ---------------------------------------------------------
MODEL_NAME = "unet_resnet34_autoencoder"
EPOCHS = 50 
RUN_PARAMS = {
    "backbone": "UNet_ResNet34_Pretrained",
    "encoder_weights": "imagenet",
    "epochs": EPOCHS,
    "encoder_lr": 1e-4,  
    "decoder_lr": 1e-3,  
    "loss_type": "Pure_SSIM_Pixels",
    "batch_size": config.BATCH_SIZE, 
    "image_size": config.IMG_SIZE,
    "seed": config.RANDOM_SEED
}

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
print(f"Starting Image SSIM Pipeline on {config.DEVICE}...")

train_loader, test_loader, normal_idx = dataloader(
    train_path=config.TRAIN_PATH, 
    test_path=config.TEST_PATH, 
    img_size=config.IMG_SIZE,
    n_train_normal=config.N_TRAIN_NORMAL,
    n_test_normal=config.N_TEST_NORMAL,
    n_test_anomaly=config.N_TEST_ANOMALY,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS
)

# ---------------------------------------------------------
# 2. Models: Pretrained U-Net Autoencoder
# ---------------------------------------------------------
print("Loading Pretrained U-Net (ResNet34) Autoencoder...")

ae_model = smp.Unet(
    encoder_name="resnet34",         
    encoder_weights="imagenet",      
    in_channels=3,                   
    classes=3,                       
    activation="sigmoid"             
).to(config.DEVICE)

ae_model = torch.compile(ae_model) 

encoder_params = list(ae_model.encoder.parameters())
decoder_params = list(ae_model.decoder.parameters()) + list(ae_model.segmentation_head.parameters())

optimizer = optim.Adam([
    {'params': encoder_params, 'lr': RUN_PARAMS["encoder_lr"]},
    {'params': decoder_params, 'lr': RUN_PARAMS["decoder_lr"]}
])

# Initialize Training Utilities
best_model_path = os.path.join(config.PROJECT_ROOT, f"{MODEL_NAME}_best.pth")
early_stopping = EarlyStopping(patience=5, path=best_model_path)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

scaler = torch.amp.GradScaler('cuda') 

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="unet_autoencoder_image")

with tracker as run:
    tracker.log_params(RUN_PARAMS)
    print("\nTraining Autoencoder with Pure SSIM Loss...")
    train_losses = []
    
    for epoch in range(EPOCHS):
        ae_model.train()
        batch_losses = []
        for imgs, _ in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]"):
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                recon = ae_model(imgs)
                ssim_loss = 1 - ssim(recon, imgs, data_range=1.0)
            
            scaler.scale(ssim_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(ssim_loss.item())
            
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)
        
        # Update Scheduler and Early Stopping
        scheduler.step(epoch_loss)
        # This triggers the __call__ method in your EarlyStopping class
        early_stopping(epoch_loss, ae_model)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss (SSIM): {epoch_loss:.6f}")
            
        if early_stopping.early_stop:
            print(f"Early Stopping triggered. Halting training.")
            break

    # ---------------------------------------------------------
    # 5. Load Best Weights & Evaluation
    # ---------------------------------------------------------
    print("\nLoading Best Weights for Final Evaluation...")
    # This loads the .pth file that EarlyStopping saved automatically
    ae_model.load_state_dict(torch.load(best_model_path)) 
    ae_model.eval()
    
    def get_ssim_errors(loader, desc):
        sample_errors = []
        labels = []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc=desc):
                imgs = imgs.to(config.DEVICE)
                recon = ae_model(imgs)
                for i in range(recon.shape[0]):
                    val = ssim(recon[i:i+1], imgs[i:i+1], data_range=1.0)
                    sample_errors.append((1 - val).item())
                labels.extend(lbls.numpy())
        return np.array(sample_errors), np.array(labels)

    train_errors, _ = get_ssim_errors(train_loader, "Train Errors")
    all_test_errors, test_labels_raw = get_ssim_errors(test_loader, "Test Errors")
    y_true_all = np.array([0 if l == normal_idx else 1 for l in test_labels_raw])

    val_errs, test_errs, val_lbls, test_lbls = train_test_split(
        all_test_errors, y_true_all, test_size=0.5, stratify=y_true_all, random_state=config.RANDOM_SEED
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
    # 6. Logging & Visuals
    # ---------------------------------------------------------
    l_p = plot_loss(train_losses, "ae_loss.png", MODEL_NAME)
    cm_p = plot_confusion_matrix(
        cm=confusion_matrix(test_lbls, test_preds), 
        target_names=['Normal', 'Anomaly'], 
        precision=precision, recall=recall, f1=f1,
        save_path="ae_cm.png", model_name=MODEL_NAME
    )
    d_p = plot_error_distribution(
        train_errors, test_errs[test_lbls==0], test_errs[test_lbls==1], 
        best_thresh, "ae_dist.png", MODEL_NAME
    )

    tracker.log_artifact(l_p)
    tracker.log_artifact(cm_p)
    tracker.log_artifact(d_p)
    tracker.log_metrics({"test_f1": float(f1), "optimal_threshold": float(best_thresh)})
    
    # ---------------------------------------------------------
    # 7. Specialized Heatmap Generator
    # ---------------------------------------------------------
# ---------------------------------------------------------
    # 7. Specialized Heatmap Generator
    # ---------------------------------------------------------
    print("\nGenerating Deep Analysis Heatmaps...")
    
    def generate_local_heatmaps(indices, title, filename):
        if len(indices) == 0: return None
        all_raw_images = []
        for imgs, _ in test_loader: all_raw_images.append(imgs)
        all_raw_tensor = torch.cat(all_raw_images)

        with torch.no_grad():
            subset_imgs = all_raw_tensor[indices].to(config.DEVICE)
            subset_recons = ae_model(subset_imgs).cpu().numpy()
            subset_origs = subset_imgs.cpu().numpy()
            subset_scores = all_test_errors[indices]
            
        num_imgs = len(indices)
        fig, axes = plt.subplots(num_imgs, 3, figsize=(15, 5 * num_imgs))
        if num_imgs == 1: axes = [axes]
            
        for i in range(num_imgs):
            orig = np.transpose(subset_origs[i], (1, 2, 0))
            recon = np.transpose(subset_recons[i], (1, 2, 0))
            heatmap = np.mean(np.abs(orig - recon), axis=-1) 
            
            axes[i][0].imshow(np.clip(orig, 0, 1))
            axes[i][0].axis('off')
            axes[i][1].imshow(np.clip(recon, 0, 1))
            axes[i][1].axis('off')
            sns.heatmap(heatmap, ax=axes[i][2], cmap='rocket', cbar=True)
            axes[i][2].axis('off')
            
        plt.tight_layout()
        target_dir = os.path.join(config.GRAPHS_DIR, MODEL_NAME)
        os.makedirs(target_dir, exist_ok=True)
        final_path = os.path.join(target_dir, filename)
        plt.savefig(final_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return final_path

    # Identify indices where the true label is Anomaly (1)
    anomaly_indices = np.where(y_true_all == 1)[0]
    anom_scores = all_test_errors[anomaly_indices]
    
    # Sort anomaly scores
    sorted_idx = np.argsort(anom_scores)
    
    # 10 Best: Highest scores (Top of the list, reversed)
    top_hits = anomaly_indices[sorted_idx[-10:][::-1]]
    
    # 10 Worst: Lowest scores (Bottom of the list)
    top_miss = anomaly_indices[sorted_idx[:10]]
    
    # Generate and log both sets
    path_hits = generate_local_heatmaps(top_hits, "Top 10 Correct Identifications", "heatmaps_hits.png")
    path_miss = generate_local_heatmaps(top_miss, "Top 10 Worst Misses (False Negatives)", "heatmaps_misses.png")
    
    tracker.log_artifact(path_hits)
    tracker.log_artifact(path_miss)
    
    print("\n" + "="*30)
    print("FINAL TEST PERFORMANCE (Using Best Checkpoint)")
    print(classification_report(test_lbls, test_preds, target_names=['Normal', 'Anomaly']))