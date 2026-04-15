import os
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


# Custom Imports
from src import config
from src.dataLoader.data_loader import dataloader

from src.helper.visualization_helper import (
    plot_loss,
    plot_confusion_matrix,
    plot_error_distribution
)
from src.helper.mlflow_helper import MLFlowTracker
from src.helper.early_stopping import EarlyStopping

from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------
# 0. Configuration & MLP Definition
# ---------------------------------------------------------
MODEL_NAME = "unet_resnet34_latent_mlp_stavya"
EPOCHS = 50
RUN_PARAMS = {
    "backbone": "ResNet34",
    "epochs": EPOCHS,
    "encoder_lr": 1e-4,  
    "decoder_lr": 1e-3,  
    "batch_size": config.BATCH_SIZE,
    "n_train": config.N_TRAIN_NORMAL
}

class LatentMLP(nn.Module):
    """A simple Multi-Layer Perceptron to classify 1D Latent Embeddings"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
        
    def train_step(self, features, labels, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        preds = self.forward(features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
train_loader, test_loader, normal_idx = dataloader(
    train_path=config.TRAIN_PATH, 
    test_path=config.TEST_PATH, 
    img_size=config.IMG_SIZE,
    n_train_normal= config.N_TRAIN_NORMAL, 
    n_test_normal=config.N_TEST_NORMAL, 
    n_test_anomaly_per_class=config.N_TEST_ANOMALY_PER_CLASS,   
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS
)

# ---------------------------------------------------------
# 2. Model Initialization
# ---------------------------------------------------------
ae_model = smp.Unet(
    encoder_name="resnet34", encoder_weights="imagenet",
    in_channels=1, classes=1, activation="sigmoid"
).to(config.DEVICE)

optimizer = optim.Adam([
    {'params': ae_model.encoder.parameters(), 'lr': RUN_PARAMS["encoder_lr"]},
    {'params': list(ae_model.decoder.parameters()) + list(ae_model.segmentation_head.parameters()), 'lr': RUN_PARAMS["decoder_lr"]}
])

early_stopping = EarlyStopping(patience=10, path=os.path.join(config.PROJECT_ROOT, f"{MODEL_NAME}.pth"))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
scaler = torch.amp.GradScaler('cuda') 

# ---------------------------------------------------------
# 3. Phase 1: Autoencoder Training
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="unet_autoencoder_latent_mlp")
best_model_vram = None

with tracker:
    tracker.log_params(RUN_PARAMS)
    train_losses = []
    
    for epoch in range(EPOCHS):
        ae_model.train()
        batch_losses = []
        for imgs, _ in tqdm(train_loader, desc=f"AE Epoch {epoch+1}"):
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                recon = ae_model(imgs)
                loss = 1 - ssim(recon, imgs, data_range=1.0)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(loss.item())
            
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        early_stopping(epoch_loss, ae_model)

        if early_stopping.counter == 0:
            best_model_vram = {k: v.clone() for k, v in ae_model.state_dict().items()}
        if early_stopping.early_stop: break

    # ---------------------------------------------------------
    # 4. Phase 2: Latent Feature Extraction
    # ---------------------------------------------------------
    ae_model.load_state_dict(best_model_vram)
    ae_model.eval()
    
    def get_features_and_maps(loader):
        origs, recons, maps, latents, folder_ids = [], [], [], [], []
        # Adaptive pooling to flatten spatial dimensions of the deepest feature map
        pool = nn.AdaptiveAvgPool2d((1, 1))
        
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc="Extracting Features & Maps"):
                imgs_device = imgs.to(config.DEVICE)
                
                # Extract the deepest latent embedding from the encoder
                encoder_features = ae_model.encoder(imgs_device)
                deepest_feature = encoder_features[-1] 
                latent_vector = pool(deepest_feature).flatten(1) # Shape: [Batch, 512]
                
                # Get full reconstruction for visualization purposes
                recon = ae_model(imgs_device)
                diff = torch.abs(imgs_device - recon)
                
                latents.append(latent_vector.cpu())
                origs.append(imgs.cpu())
                recons.append(recon.cpu())
                maps.append(diff.cpu())
                folder_ids.extend(lbls.numpy())
                
        return torch.cat(latents), torch.cat(origs), torch.cat(recons), torch.cat(maps), np.array(folder_ids)
    print("Test loader dataset size:", len(test_loader.dataset))
    all_latents, all_origs, all_recons, all_maps, all_folder_labels = get_features_and_maps(test_loader)
    all_binary_labels = np.array([0 if l == normal_idx else 1 for l in all_folder_labels])

    # Ensure all parallel arrays are split correctly
    (val_latents, test_latents, val_maps, test_maps, val_bin, test_bin, val_raw, test_raw, val_origs, test_origs, val_recons, test_recons) = train_test_split(
        all_latents, all_maps, all_binary_labels, all_folder_labels, all_origs, all_recons, test_size=0.5, random_state=42, stratify=all_binary_labels
    )

    mlp = LatentMLP(input_dim=val_latents.shape[1]).to(config.DEVICE)
    mlp_optimizer = optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_weight = (len(val_bin) - np.sum(val_bin)) / np.sum(val_bin)

    mlp_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=config.DEVICE)
    )  
    mlp_loader = DataLoader(
        TensorDataset(val_latents, torch.tensor(val_bin).float()), 
        batch_size=64, shuffle=True
    )

    print("Training MLP on Latent Embeddings (Imbalanced)...")
    for _ in range(30): # MLP trains much faster, 30 epochs is safe and quick
        for x, l in mlp_loader:
            mlp.train_step(x.to(config.DEVICE), l.to(config.DEVICE), mlp_optimizer, mlp_criterion)

    # ---------------------------------------------------------
    # 5. Final Metrics & Thresholding
    # ---------------------------------------------------------
    
    def get_mlp_preds(model, features):
        model.eval()
        loader = DataLoader(TensorDataset(features), batch_size=128, shuffle=False)
        probs = []
        with torch.no_grad():
            for (x,) in loader:
                logits = model(x.to(config.DEVICE))
                probs.extend(torch.sigmoid(logits).cpu().numpy())
        return np.array(probs)

    v_combined = get_mlp_preds(mlp, val_latents)
    t_combined = get_mlp_preds(mlp, test_latents)
    # ---------------------------------------------------------
    # Reconstruction error (mean per image)
    # ---------------------------------------------------------

    v_recon_error = val_maps.view(val_maps.size(0), -1).mean(dim=1).numpy()
    t_recon_error = test_maps.view(test_maps.size(0), -1).mean(dim=1).numpy()
    # ---------------------------------------------------------
    # Normalize using VALIDATION stats (important)
    # ---------------------------------------------------------

    def normalize_with_ref(x, ref_min, ref_max):
        x_norm = (x - ref_min) / (ref_max - ref_min + 1e-8)
        return np.clip(x_norm, 0.0, 1.0)

    # latent scores
    v_latent = v_combined
    t_latent = t_combined

    # get min/max from validation ONLY
    latent_min, latent_max = v_latent.min(), v_latent.max()
    recon_min, recon_max = v_recon_error.min(), v_recon_error.max()

    # normalize both val and test using SAME stats
    v_latent_norm = normalize_with_ref(v_latent, latent_min, latent_max)
    t_latent_norm = normalize_with_ref(t_latent, latent_min, latent_max)

    v_recon_norm = normalize_with_ref(v_recon_error, recon_min, recon_max)
    t_recon_norm = normalize_with_ref(t_recon_error, recon_min, recon_max)

    # ---------------------------------------------------------
    # Combined score
    # ---------------------------------------------------------

    alpha = 0.5

    v_combined = alpha * v_latent_norm + (1 - alpha) * v_recon_norm
    t_combined = alpha * t_latent_norm + (1 - alpha) * t_recon_norm
    # Threshold-independent metrics
    # ---------------------------------------------------------
    roc_auc = roc_auc_score(test_bin, t_combined)
    pr_auc = average_precision_score(test_bin, t_combined)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    # # Calibrate threshold on validation set
    # thresholds = np.linspace(v_probs.min(), v_probs.max(), 100)
    # best_f1, opt_thresh = 0, 0.5
    # for t in thresholds:
    #     f1_score = precision_recall_fscore_support(val_bin, (v_probs > t).astype(int), average='binary', zero_division=0)[2]
    #     if f1_score > best_f1: 
    #         best_f1, opt_thresh = f1_score, t

    # final_preds = (t_probs > opt_thresh).astype(int)

   # ---------------------------------------------------------
    # Thresholding (Improved)
    # ---------------------------------------------------------
    thresholds = np.linspace(0, 1, 200)
    target_precision = 0.90

    best_f1 = -1
    pc_thresh = None

    fallback_best_f1 = -1
    fallback_thresh = 0.5

    for t in thresholds:
        preds = (v_combined > t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            val_bin, preds, average='binary', zero_division=0
        )

        # always track best F1 (fallback)
        if f1 > fallback_best_f1:
            fallback_best_f1 = f1
            fallback_thresh = t

        # precision-constrained
        if p >= target_precision and f1 > best_f1:
            best_f1 = f1
            pc_thresh = t

    # fallback if no threshold meets precision
    if pc_thresh is None:
        pc_thresh = fallback_thresh

    # --- 2. Percentile threshold ---
    perc_thresh = np.percentile(v_combined[val_bin == 0], 95)

    # Predictions
    pc_preds = (t_combined > pc_thresh).astype(int)
    perc_preds = (t_combined > perc_thresh).astype(int)

    # Use precision-constrained as main
    final_preds = pc_preds
    opt_thresh = pc_thresh  # keep compatibility with rest of code
    # ---------------------------------------------------------
    # 6. Visualization & Reporting
    # ---------------------------------------------------------
    p, r, f, _ = precision_recall_fscore_support(test_bin, final_preds, average='binary')
    
    # Evaluate percentile threshold
    perc_p, perc_r, perc_f, _ = precision_recall_fscore_support(
        test_bin, perc_preds, average='binary', zero_division=0
    )

    print("\n--- Threshold Comparison ---")
    print(f"Precision-Constrained → P:{p:.3f}, R:{r:.3f}, F1:{f:.3f}, thresh:{pc_thresh:.4f}")
    print(f"Percentile (95)      → P:{perc_p:.3f}, R:{perc_r:.3f}, F1:{perc_f:.3f}, thresh:{perc_thresh:.4f}")
    
    cm = confusion_matrix(test_bin, final_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    tracker.log_artifact(plot_loss(train_losses, "loss.png", MODEL_NAME))
    tracker.log_artifact(plot_confusion_matrix(
        cm=cm,
        target_names=['Normal', 'Anomaly'],
        precision=p,
        recall=r,
        f1=f,
        specificity=specificity,
        auc_roc=roc_auc,
        auc_pr=pr_auc,
        save_path="cm.png",
        model_name=MODEL_NAME
    ))
    tracker.log_artifact(plot_error_distribution(
        train_errors=v_combined[val_bin == 0], 
        test_normal_errors=t_combined[test_bin == 0], 
        test_anomaly_errors=t_combined[test_bin == 1],
        threshold=opt_thresh,
        save_path="mlp_dist.png", 
        model_name=MODEL_NAME
    ))

    # --- Sample Visualization Function ---
    def save_sample_visualization(orig, recon, diff, true_bin, pred_bin, score, class_id, save_name):
        save_dir = os.path.join(config.GRAPHS_DIR, MODEL_NAME, "samples")
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(orig.squeeze().numpy(), cmap='gray')
        axes[0].set_title("Original OCT")
        axes[0].axis('off')
        
        axes[1].imshow(recon.squeeze().numpy(), cmap='gray')
        axes[1].set_title("Output (Reconstructed) OCT")
        axes[1].axis('off')
        
        sns.heatmap(diff.squeeze().numpy(), cmap='jet', ax=axes[2], cbar=True)
        axes[2].set_title(f"Difference Heatmap\nScore: {score:.4f}")
        axes[2].axis('off')
        
        status = "Correct" if true_bin == pred_bin else "Incorrect"
        fig.suptitle(f"Original Class ID: {int(class_id)} | True Binary: {true_bin} | Pred Binary: {pred_bin} ({status})", fontsize=16)
        
        plt.tight_layout()
        full_path = os.path.join(save_dir, save_name)
        plt.savefig(full_path)
        plt.close(fig)
        return full_path

    # --- 1. Normal predicted as Normal ---
    idx_tn = np.where((test_bin == 0) & (final_preds == 0))[0]
    if len(idx_tn) > 0:
        i = idx_tn[0]
        tracker.log_artifact(save_sample_visualization(test_origs[i], test_recons[i], test_maps[i], test_bin[i], final_preds[i], t_combined[i], test_raw[i], "normal_pred_normal.png"))

    # --- 2. Normal predicted as Anomaly ---
    idx_fp = np.where((test_bin == 0) & (final_preds == 1))[0]
    if len(idx_fp) > 0:
        i = idx_fp[0]
        tracker.log_artifact(save_sample_visualization(test_origs[i], test_recons[i], test_maps[i], test_bin[i], final_preds[i], t_combined[i], test_raw[i], "normal_pred_anomaly.png"))

    # --- 3. Worst 5 Disease predicted as Normal (False Negatives) ---
    idx_fn = np.where((test_bin == 1) & (final_preds == 0))[0]
    if len(idx_fn) > 0:
        worst_fn_indices = idx_fn[np.argsort(t_combined[idx_fn])][:5]
        for rank, i in enumerate(worst_fn_indices):
            tracker.log_artifact(save_sample_visualization(
                test_origs[i], test_recons[i], test_maps[i], 
                test_bin[i], final_preds[i], t_combined[i], test_raw[i], 
                f"worst_fn_{rank+1}_class_{int(test_raw[i])}.png"
            ))

    # --- Image of the Table Breakdown ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    unique_classes = np.unique(test_raw)
    table_data = []
    for cls in unique_classes:
        mask = (test_raw == cls)
        preds_for_cls = final_preds[mask]
        
        if cls == normal_idx:
            true_type = "Normal"
            correct = np.sum(preds_for_cls == 0)
            wrong = np.sum(preds_for_cls == 1)
        else:
            true_type = "Disease"
            correct = np.sum(preds_for_cls == 1)
            wrong = np.sum(preds_for_cls == 0)
            
        table_data.append([f"Class {int(cls)}", true_type, correct, wrong])
        
    table = ax.table(cellText=table_data, colLabels=['Original Class', 'True Type', 'Correct Predictions', 'Wrong Predictions'], loc='center', cellLoc='center')
    table.scale(1, 1.8)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    
    plt.title("Prediction Breakdown by Category (Latent MLP)", fontsize=16, pad=20)
    
    save_dir = os.path.join(config.GRAPHS_DIR, MODEL_NAME)
    os.makedirs(save_dir, exist_ok=True)
    table_path = os.path.join(save_dir, "prediction_breakdown_table.png")
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()
    tracker.log_artifact(table_path)

    # --------------------------------------------------
    tracker.log_metrics({
        "pc_f1": float(f),
        "pc_precision": float(p),
        "pc_recall": float(r),
        "pc_threshold": float(pc_thresh),

        "perc_f1": float(perc_f),
        "perc_precision": float(perc_p),
        "perc_recall": float(perc_r),
        "perc_threshold": float(perc_thresh),

        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    })    
    print("Total extracted:", len(all_binary_labels))
    print("Normals extracted:", np.sum(all_binary_labels == 0))
    print("Anomalies extracted:", np.sum(all_binary_labels == 1))
    print("Final eval size:", len(test_bin))
    print("Final eval normals:", np.sum(test_bin == 0))
    print("Final eval anomalies:", np.sum(test_bin == 1))
    print("="*30 + "\nOVERALL REPORT\n" + classification_report(test_bin, final_preds, target_names=['Normal', 'Anomaly']))