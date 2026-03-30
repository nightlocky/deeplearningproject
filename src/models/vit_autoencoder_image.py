import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim

# Import Config and Helpers
import config
from dataLoader.dataLoader import dataloader
from helper.visualization_helper import (
    plot_loss, 
    plot_error_distribution, 
    plot_confusion_matrix, 
    plot_anomaly_comparison  # <--- ONLY import the plotting tool, not the generator
)
from helper.mlflow_helper import MLFlowTracker

MODEL_NAME = "image_autoencoder"
EPOCHS = 20
RUN_PARAMS = {
    "backbone": "Convolutional_Image_AE",
    "encoder_params": "Conv2d[32, 64, 128]",
    "decoder_params": "ConvTranspose2d[64, 32, 3]",
    "epochs": EPOCHS,
    "learning_rate": 0.001,
    "loss_type": "Pure_SSIM_Pixels",
    "batch_size": config.BATCH_SIZE, 
    "image_size": config.IMG_SIZE,
    "seed": config.RANDOM_SEED
}

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
# 2. Models: Image Autoencoder
# ---------------------------------------------------------
print("Loading Convolutional Autoencoder...")

class ImageAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae_model = ImageAutoencoder().to(config.DEVICE)
ae_model = torch.compile(ae_model) 

optimizer = optim.Adam(ae_model.parameters(), lr=RUN_PARAMS["learning_rate"])
scaler = torch.amp.GradScaler('cuda') 

# ---------------------------------------------------------
# 4. Training Loop (Pure SSIM Loss on Pixels)
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="vit_autoencoder_image")

with tracker as run:
    tracker.log_params(RUN_PARAMS)
    print("\nTraining Autoencoder with Pure SSIM Loss...")
    train_losses = []
    
    for epoch in range(EPOCHS):
        ae_model.train()
        batch_losses = []
        for imgs, _ in train_loader:
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                recon = ae_model(imgs)
                ssim_loss = 1 - ssim(recon, imgs, data_range=1.0)
            
            scaler.scale(ssim_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(ssim_loss.item())
            
        train_losses.append(np.mean(batch_losses))
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss (SSIM): {train_losses[-1]:.6f}")

    # ---------------------------------------------------------
    # 5. Threshold Optimization
    # ---------------------------------------------------------
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

    print("\nCalculating SSIM Errors...")
    train_errors, _ = get_ssim_errors(train_loader, "Train Set")
    all_test_errors, test_labels_raw = get_ssim_errors(test_loader, "Test Set")
    y_true_all = np.array([0 if l == normal_idx else 1 for l in test_labels_raw])

    val_errs, test_errs, val_lbls, test_lbls = train_test_split(
        all_test_errors, y_true_all, 
        test_size=0.5, 
        stratify=y_true_all, 
        random_state=config.RANDOM_SEED
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
    # 6. Logging & Standard Visuals
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

    tracker.log_metrics({
        "precision": float(precision), 
        "recall": float(recall), 
        "f1": float(f1),
        "threshold": float(best_thresh)
    })
    
    # ---------------------------------------------------------
    # 7. Deep Analysis: Model-Specific Heatmap Generation
    # ---------------------------------------------------------
    print("\nGenerating Deep Analysis Heatmaps...")
    
    # 1. Collect all raw images
    all_raw_images = []
    for imgs, _ in test_loader:
        all_raw_images.append(imgs)
    all_raw_tensor = torch.cat(all_raw_images)
    
    # 2. Identify Top 10 Hits and Worst 10 Misses
    anomaly_indices = np.where(y_true_all == 1)[0]
    anom_scores = all_test_errors[anomaly_indices]
    
    top_hits = anomaly_indices[np.argsort(anom_scores)[-10:][::-1]]
    top_miss = anomaly_indices[np.argsort(anom_scores)[:10]]

    # 3. Define the local generator tailored explicitly for THIS image model
    def generate_local_heatmaps(indices, title, filename):
        if len(indices) == 0: return None
        
        ae_model.eval()
        with torch.no_grad():
            # OPTIMIZATION: Extract only the 10 images we need, saving massive GPU RAM
            subset_imgs = all_raw_tensor[indices].to(config.DEVICE)
            subset_recons = ae_model(subset_imgs).cpu().numpy()
            subset_origs = subset_imgs.cpu().numpy()
            subset_scores = all_test_errors[indices]
            
            # Since we sliced the array, the new indices for the plotter are simply 0 to N
            local_indices = np.arange(len(indices))
            
            # Pass to the universal drawing tool
            return plot_anomaly_comparison(
                orig_imgs=subset_origs, 
                recon_imgs=subset_recons, 
                scores=subset_scores, 
                indices=local_indices, 
                title=title, 
                filename=filename, 
                model_name=MODEL_NAME
            )

    # 4. Generate and Log
    path_hits = generate_local_heatmaps(top_hits, "Top 10 Correctly Identified Anomalies", "heatmaps_best_hits.png")
    path_miss = generate_local_heatmaps(top_miss, "Top 10 Missed Anomalies (False Negatives)", "heatmaps_worst_misses.png")
    
    tracker.log_artifact(path_hits)
    tracker.log_artifact(path_miss)
    
    torch.save(ae_model.state_dict(), os.path.join(config.PROJECT_ROOT, "image_ae_model.pth"))
    print("\n" + "="*30)
    print(classification_report(test_lbls, test_preds, target_names=['Normal', 'Anomaly']))