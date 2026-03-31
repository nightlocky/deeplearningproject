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
from helper.EarlyStopping import EarlyStopping 

# ---------------------------------------------------------
# 0. Global Settings & Params
# ---------------------------------------------------------
MODEL_NAME = "unet_resnet34_autoencoder_1ch"
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
    "seed": config.RANDOM_SEED,
    "channels": 1 # Tracking that we are now in 1-channel mode
}

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
print(f"Starting Image SSIM Pipeline (1-Channel) on {config.DEVICE}...")

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
# 2. Models: Pretrained U-Net Autoencoder (Modified for 1-Channel)
# ---------------------------------------------------------
print("Loading Pretrained U-Net (ResNet34) Autoencoder...")

ae_model = smp.Unet(
    encoder_name="resnet34",         
    encoder_weights="imagenet",      
    in_channels=1,     # Changed from 3 to 1                
    classes=1,         # Changed from 3 to 1                      
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
tracker = MLFlowTracker(experiment_name="unet_autoencoder_image_1ch")

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
        
        scheduler.step(epoch_loss)
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
    
    # 7. Five-Category Deep Analysis (Corrected Indexing)
    # ---------------------------------------------------------
    print("\nGenerating Five-Category Deep Analysis (10 samples each)...")
    all_preds = (all_test_errors > best_thresh).astype(int)

    def generate_category_heatmaps(indices, category_name, filename):
        if len(indices) == 0: 
            print(f"No samples found for category: {category_name}")
            return None
            
        selected_indices = indices[:10]
        num_imgs = len(selected_indices)
        
        # Memory-efficient way to get specific tensors from the loader
        all_raw_images = []
        for imgs, _ in test_loader: 
            all_raw_images.append(imgs)
        all_raw_tensor = torch.cat(all_raw_images)

        with torch.no_grad():
            subset_imgs = all_raw_tensor[selected_indices].to(config.DEVICE)
            subset_recons = ae_model(subset_imgs).cpu().numpy()
            subset_origs = subset_imgs.cpu().numpy()
            # Use the full error array for scoring
            subset_scores = all_test_errors[selected_indices]
            
        fig, axes = plt.subplots(num_imgs, 3, figsize=(15, 5 * num_imgs))
        if num_imgs == 1: axes = np.expand_dims(axes, axis=0)
            
        fig.suptitle(f"Category: {category_name}\nThreshold: {best_thresh:.4f}", fontsize=16)

        for i in range(num_imgs):
            orig = np.squeeze(subset_origs[i])
            recon = np.squeeze(subset_recons[i])
            # We use L1 (absolute difference) for the heatmap visualization
            heatmap = np.abs(orig - recon) 
            score = subset_scores[i]
            
            # Column 1: Original
            axes[i][0].imshow(orig, cmap='gray')
            axes[i][0].set_title(f"Original (Error: {score:.4f})")
            axes[i][0].axis('off')
            
            # Column 2: Reconstruction
            axes[i][1].imshow(recon, cmap='gray')
            axes[i][1].set_title("Reconstruction")
            axes[i][1].axis('off')
            
            # Column 3: Error Heatmap (Rocket highlights where the model 'missed')
            sns.heatmap(heatmap, ax=axes[i][2], cmap='rocket', cbar=True)
            axes[i][2].set_title("Difference Map")
            axes[i][2].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        target_dir = os.path.join(config.GRAPHS_DIR, MODEL_NAME)
        os.makedirs(target_dir, exist_ok=True)
        final_path = os.path.join(target_dir, filename)
        plt.savefig(final_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return final_path

    # Category Logic (Based on Full 100% Test Set)
    tn_idx = np.where((y_true_all == 0) & (all_preds == 0))[0]
    fp_idx = np.where((y_true_all == 0) & (all_preds == 1))[0]
    fn_idx = np.where((y_true_all == 1) & (all_preds == 0))[0]
    
    # Get original class names from the underlying ImageFolder
    # Use .dataset.dataset because test_loader.dataset is a 'Subset'
    inv_class_to_idx = {v: k for k, v in test_loader.dataset.dataset.class_to_idx.items()}
    
    # Log Category 1 (Normal Correct) and 2 (Normal False Alarm)
    tracker.log_artifact(generate_category_heatmaps(tn_idx, "TN: Normal correctly identified", "cat1_tn.png"))
    tracker.log_artifact(generate_category_heatmaps(fp_idx, "FP: Normal flagged as Anomaly", "cat2_fp.png"))

    # Log Categories 3-5 (The specific Anomaly Misses)
    for class_idx, class_name in inv_class_to_idx.items():
        if class_name == 'NORMAL': continue
        
        # Logic: Label is this specific disease AND model predicted it was Normal
        specific_fn_idx = [i for i in fn_idx if test_labels_raw[i] == class_idx]
        
        filename = f"cat_fn_{class_name.lower()}.png"
        path = generate_category_heatmaps(specific_fn_idx, f"FN: {class_name} missed by model", filename)
        if path: tracker.log_artifact(path)

    print("\n" + "="*30)
    print("FINAL TEST PERFORMANCE (Using Best Checkpoint)")
    print(classification_report(test_lbls, test_preds, target_names=['Normal', 'Anomaly']))