import os
import gc
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
from src import config_tune  
from src.dataLoader.dataLoader import dataloader
from src.helper.visualization_helper import plot_loss, plot_confusion_matrix, plot_error_distribution
from src.helper.mlflow_helper import MLFlowTracker
from src.helper.EarlyStopping import EarlyStopping 
from latent_mlp import LatentMLP

# ---------------------------------------------------------
# 1. Phase 1: Train Autoencoder
# ---------------------------------------------------------
def train_ae(backbone, loss_alpha, run_name, train_loader):
    print(f"\n--- Training Autoencoder: {run_name} ---")
    run_dir = os.path.join(config.TUNING_DIR, config_tune.EXPERIMENT_NAME, run_name)
    os.makedirs(run_dir, exist_ok=True)
    model_save_path = os.path.join(run_dir, f"{run_name}_best_ae.pth")

    ae_model = smp.Unet(
        encoder_name=backbone, encoder_weights="imagenet",
        in_channels=1, classes=1, activation="sigmoid"
    ).to(config.DEVICE)

    optimizer = optim.Adam([
        {'params': ae_model.encoder.parameters(), 'lr': config_tune.ENCODER_LR},
        {'params': list(ae_model.decoder.parameters()) + list(ae_model.segmentation_head.parameters()), 'lr': config_tune.DECODER_LR}
    ])

    early_stopping = EarlyStopping(patience=10, path=model_save_path)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda') 
    l1_criterion = nn.L1Loss() 

    train_losses = []
    
    for epoch in range(config_tune.TUNING_EPOCHS):
        ae_model.train()
        batch_losses = []
        for imgs, _ in tqdm(train_loader, desc=f"AE Epoch {epoch+1}"):
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                recon = ae_model(imgs)
                loss_ssim = 1 - ssim(recon, imgs, data_range=1.0)
                loss_l1 = l1_criterion(recon, imgs)
                loss = (loss_alpha * loss_ssim) + ((1.0 - loss_alpha) * loss_l1)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(loss.item())
            
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        early_stopping(epoch_loss, ae_model)

        if early_stopping.early_stop: 
            break
            
    plot_loss(train_losses, os.path.join(run_dir, "loss.png"), run_name)
    del ae_model, optimizer
    torch.cuda.empty_cache()
    return model_save_path

# ---------------------------------------------------------
# 2. Phase 2: Feature Extraction
# ---------------------------------------------------------
def extract_features(backbone, model_path, test_loader, normal_idx):
    print("\n--- Extracting Features ---")
    ae_model = smp.Unet(
        encoder_name=backbone, encoder_weights=None, 
        in_channels=1, classes=1, activation="sigmoid"
    ).to(config.DEVICE)
    
    ae_model.load_state_dict(torch.load(model_path, weights_only=True))
    ae_model.eval()

    origs, recons, maps, latents, folder_ids = [], [], [], [], []
    pool = nn.AdaptiveAvgPool2d((1, 1))
    
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Extracting"):
            imgs_device = imgs.to(config.DEVICE)
            encoder_features = ae_model.encoder(imgs_device)
            latent_vector = pool(encoder_features[-1]).flatten(1)
            recon = ae_model(imgs_device)
            diff = torch.abs(imgs_device - recon)
            
            latents.append(latent_vector.cpu())
            origs.append(imgs.cpu())
            recons.append(recon.cpu())
            maps.append(diff.cpu())
            folder_ids.extend(lbls.numpy())
            
    all_latents, all_origs, all_recons, all_maps, all_folder_labels = torch.cat(latents), torch.cat(origs), torch.cat(recons), torch.cat(maps), np.array(folder_ids)
    all_binary_labels = np.array([0 if l == normal_idx else 1 for l in all_folder_labels])

    del ae_model
    torch.cuda.empty_cache()

    return train_test_split(
        all_latents, all_maps, all_binary_labels, all_folder_labels, all_origs, all_recons, 
        test_size=0.5, stratify=all_binary_labels
    )

# ---------------------------------------------------------
# 3. Phase 3: Train & Evaluate MLP
# ---------------------------------------------------------
def train_and_evaluate_mlp(data_splits, hidden_layers, dropout, run_name, normal_idx, tracker):
    print(f"\n--- Evaluating MLP: {run_name} ---")
    (val_latents, test_latents, val_maps, test_maps, val_bin, test_bin, val_raw, test_raw, val_origs, test_origs, val_recons, test_recons) = data_splits
    
    run_dir = os.path.join(config.TUNING_DIR, config_tune.EXPERIMENT_NAME, run_name)
    os.makedirs(run_dir, exist_ok=True)
    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    mlp = LatentMLP(input_dim=val_latents.shape[1], hidden_layers=hidden_layers, dropout_rate=dropout).to(config.DEVICE)
    mlp_optimizer = optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    mlp_criterion = nn.BCELoss()
    
    mlp_loader = DataLoader(TensorDataset(val_latents, torch.tensor(val_bin).float()), batch_size=64, shuffle=True)

    for _ in range(config_tune.MLP_EPOCHS):
        for x, l in mlp_loader:
            mlp.train_step(x.to(config.DEVICE), l.to(config.DEVICE), mlp_optimizer, mlp_criterion)

    def get_mlp_preds(model, features):
        model.eval()
        loader = DataLoader(TensorDataset(features), batch_size=128, shuffle=False)
        probs = []
        with torch.no_grad():
            for (x,) in loader:
                probs.extend(model(x.to(config.DEVICE)).cpu().numpy())
        return np.array(probs)

    v_probs = get_mlp_preds(mlp, val_latents)
    t_probs = get_mlp_preds(mlp, test_latents)

    thresholds = np.linspace(v_probs.min(), v_probs.max(), 100)
    best_f1, opt_thresh = 0, 0.5
    for t in thresholds:
        f1_score = precision_recall_fscore_support(val_bin, (v_probs > t).astype(int), average='binary', zero_division=0)[2]
        if f1_score > best_f1: 
            best_f1, opt_thresh = f1_score, t

    final_preds = (t_probs > opt_thresh).astype(int)
    p, r, final_test_f1, _ = precision_recall_fscore_support(test_bin, final_preds, average='binary')
    
    # --- Artifact Logging ---
    with tracker:
        tracker.log_artifact(plot_confusion_matrix(confusion_matrix(test_bin, final_preds), ['Normal', 'Anomaly'], p, r, final_test_f1, os.path.join(run_dir, "cm.png"), run_name))
        tracker.log_artifact(plot_error_distribution(v_probs[val_bin == 0], t_probs[test_bin == 0], t_probs[test_bin == 1], opt_thresh, os.path.join(run_dir, "mlp_dist.png"), run_name))

        def save_sample_visualization(orig, recon, diff, true_bin, pred_bin, score, class_id, save_name):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(orig.squeeze().numpy(), cmap='gray'); axes[0].axis('off')
            axes[1].imshow(recon.squeeze().numpy(), cmap='gray'); axes[1].axis('off')
            sns.heatmap(diff.squeeze().numpy(), cmap='jet', ax=axes[2], cbar=True); axes[2].axis('off')
            status = "Correct" if true_bin == pred_bin else "Incorrect"
            fig.suptitle(f"Original Class ID: {int(class_id)} | True Binary: {true_bin} | Pred: {pred_bin} ({status})", fontsize=16)
            plt.tight_layout()
            full_path = os.path.join(samples_dir, save_name)
            plt.savefig(full_path)
            plt.close(fig)
            return full_path

        # False Negatives & Table Generation...
        idx_fn = np.where((test_bin == 1) & (final_preds == 0))[0]
        if len(idx_fn) > 0:
            for rank, i in enumerate(idx_fn[np.argsort(t_probs[idx_fn])][:5]):
                tracker.log_artifact(save_sample_visualization(test_origs[i], test_recons[i], test_maps[i], test_bin[i], final_preds[i], t_probs[i], test_raw[i], f"{run_name}_fn_{rank+1}.png"))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        table_data = [[f"Class {int(cls)}", "Normal" if cls == normal_idx else "Disease", np.sum(final_preds[test_raw == cls] == (0 if cls == normal_idx else 1)), np.sum(final_preds[test_raw == cls] == (1 if cls == normal_idx else 0))] for cls in np.unique(test_raw)]
        table = ax.table(cellText=table_data, colLabels=['Original Class', 'True Type', 'Correct', 'Wrong'], loc='center', cellLoc='center')
        table.scale(1, 1.8)
        table_path = os.path.join(run_dir, "prediction_breakdown.png")
        plt.savefig(table_path, bbox_inches='tight')
        plt.close()
        tracker.log_artifact(table_path)
        tracker.log_metrics({"test_f1": float(final_test_f1), "threshold": float(opt_thresh)})

    del mlp
    gc.collect()
    torch.cuda.empty_cache()
    return final_test_f1


# ---------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    
    print("Loading datasets...")
    train_loader, test_loader, normal_idx = dataloader(
        train_path=config.TRAIN_PATH, test_path=config.TEST_PATH, img_size=config.IMG_SIZE,
        n_train_normal=config.TEST_N_TRAIN_NORMAL, n_test_normal=config.TEST_N_TEST_NORMAL, 
        n_test_anomaly_per_class=config.TEST_N_TEST_ANOMALY_PER_CLASS, batch_size=config.BATCH_SIZE, num_workers=16
    )

    tracker = MLFlowTracker(experiment_name=config_tune.EXPERIMENT_NAME)
    results = {}
    
    # --- STEP 1: TUNE BACKBONE ---
    best_backbone, best_bb_f1 = None, -1
    for backbone in config_tune.BACKBONES:
        run_name = f"step1_bb_{backbone}"
        ae_path = train_ae(backbone, 1.0, run_name, train_loader)
        splits = extract_features(backbone, ae_path, test_loader, normal_idx)
        f1 = train_and_evaluate_mlp(splits, [256, 64], 0.3, run_name, normal_idx, tracker)
        results[run_name] = f1
        if f1 > best_bb_f1: best_backbone, best_bb_f1 = backbone, f1
            
    print(f"\n Best Backbone: {best_backbone} (F1: {best_bb_f1:.4f})\n")

    # --- STEP 2: TUNE LOSS ---
    best_alpha, best_loss_f1 = None, -1
    best_ae_path = None 
    for alpha in config_tune.LOSS_ALPHAS:
        run_name = f"step2_loss_{alpha}"
        ae_path = train_ae(best_backbone, alpha, run_name, train_loader)
        splits = extract_features(best_backbone, ae_path, test_loader, normal_idx)
        f1 = train_and_evaluate_mlp(splits, [256, 64], 0.3, run_name, normal_idx, tracker)
        results[run_name] = f1
        if f1 > best_loss_f1: 
            best_alpha, best_loss_f1 = alpha, f1
            best_ae_path = ae_path 

    print(f"\n Best Loss Alpha: {best_alpha} (F1: {best_loss_f1:.4f})\n")

    # --- STEP 3: TUNE MLP (Using best AE features) ---
    print("\n STARTING RAPID MLP TUNING")
    best_splits = extract_features(best_backbone, best_ae_path, test_loader, normal_idx)
    
    best_mlp_f1, best_mlp_config = -1, None
    
    for arch in config_tune.MLP_ARCHITECTURES:
        for drop in config_tune.MLP_DROPOUTS:
            arch_str = "_".join(map(str, arch))
            run_name = f"step3_mlp_{arch_str}_drop_{drop}"
            
            f1 = train_and_evaluate_mlp(best_splits, arch, drop, run_name, normal_idx, tracker)
            results[run_name] = f1
            
            if f1 > best_mlp_f1:
                best_mlp_f1, best_mlp_config = f1, (arch, drop)

    print("\n" + "="*50)
    print(" FULL TUNING COMPLETE")
    print(f"Optimal Backbone: {best_backbone}")
    print(f"Optimal Loss Alpha: {best_alpha}")
    print(f"Optimal MLP: {best_mlp_config[0]} | Dropout: {best_mlp_config[1]}")
    print("="*50)