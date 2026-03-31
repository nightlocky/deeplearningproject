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
from src.dataLoader.dataLoader import dataloader
from src.helper.visualization_helper import (
    plot_loss,
    plot_confusion_matrix,
    plot_error_distribution
)
from src.helper.mlflow_helper import MLFlowTracker
from src.helper.EarlyStopping import EarlyStopping 
from src.models.errorCNNClassifier import ErrorCNN
# ---------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------
MODEL_NAME = "unet_resnet34_autoencoder"
EPOCHS = 5 
RUN_PARAMS = {
    "backbone": "ResNet34",
    "epochs": EPOCHS,
    "encoder_lr": 1e-4,  
    "decoder_lr": 1e-3,  
    "batch_size": config.BATCH_SIZE,
    "n_train": config.N_TRAIN_NORMAL
}

# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
train_loader, test_loader, normal_idx = dataloader(
    train_path=config.TRAIN_PATH, 
    test_path=config.TEST_PATH, 
    img_size=config.IMG_SIZE,
    n_train_normal=100, 
    n_test_normal=50, 
    n_test_anomaly=5,   
    batch_size=config.BATCH_SIZE,
    num_workers=0
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

early_stopping = EarlyStopping(patience=15, path=os.path.join(config.PROJECT_ROOT, f"{MODEL_NAME}.pth"))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
scaler = torch.amp.GradScaler('cuda') 

# ---------------------------------------------------------
# 3. Phase 1: Autoencoder Training
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="unet_autoencoder_image_withCNN")
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
    # 4. Phase 2: CNN Spatial Thresholding
    # ---------------------------------------------------------
    ae_model.load_state_dict(best_model_vram)
    ae_model.eval()
    
    def get_spatial_errors(loader):
        maps, folder_ids = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc="Extracting Maps"):
                diff = torch.abs(imgs.to(config.DEVICE) - ae_model(imgs.to(config.DEVICE)))
                maps.append(diff.cpu())
                folder_ids.extend(lbls.numpy())
        return torch.cat(maps), np.array(folder_ids)

    all_maps, all_folder_labels = get_spatial_errors(test_loader)
    all_binary_labels = np.array([0 if l == normal_idx else 1 for l in all_folder_labels])

    (val_maps, test_maps, val_bin, test_bin, val_raw, test_raw) = train_test_split(
        all_maps, all_binary_labels, all_folder_labels, test_size=0.5, stratify=all_binary_labels
    )

    cnn = ErrorCNN().to(config.DEVICE)
    cnn_optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
    cnn_criterion = nn.BCELoss()
    
    cnn_loader = DataLoader(
        TensorDataset(val_maps, torch.tensor(val_bin).float()), 
        batch_size=32, shuffle=True
    )

    print("Training CNN Spatial Threshold...")
    for _ in range(15):
        for m, l in cnn_loader:
            # Matches the train_step(self, maps, labels, optimizer, criterion) definition
            cnn.train_step(m.to(config.DEVICE), l.to(config.DEVICE), cnn_optimizer, cnn_criterion)

    # ---------------------------------------------------------
    # 5. Final Metrics & Thresholding
    # ---------------------------------------------------------
    cnn.eval()
    with torch.no_grad():
        v_probs = cnn(val_maps.to(config.DEVICE)).cpu().numpy()
        t_probs = cnn(test_maps.to(config.DEVICE)).cpu().numpy()

    thresholds = np.linspace(v_probs.min(), v_probs.max(), 100)
    best_f1, opt_thresh = 0, 0.5
    for t in thresholds:
        f1_score = precision_recall_fscore_support(val_bin, (v_probs > t).astype(int), average='binary', zero_division=0)[2]
        if f1_score > best_f1: 
            best_f1, opt_thresh = f1_score, t

    final_preds = (t_probs > opt_thresh).astype(int)

    # ---------------------------------------------------------
    # 6. Visualization & Reporting
    # ---------------------------------------------------------
    p, r, f, _ = precision_recall_fscore_support(test_bin, final_preds, average='binary')
    
    tracker.log_artifact(plot_loss(train_losses, "loss.png", MODEL_NAME))
    tracker.log_artifact(plot_confusion_matrix(confusion_matrix(test_bin, final_preds), ['Normal', 'Anomaly'], p, r, f, "cm.png", MODEL_NAME))
    tracker.log_artifact(plot_error_distribution(
        train_errors=v_probs[val_bin == 0], 
        test_normal_errors=t_probs[test_bin == 0], 
        test_anomaly_errors=t_probs[test_bin == 1],
        threshold=opt_thresh,
        save_path="cnn_dist.png", 
        model_name=MODEL_NAME
    ))
    
    tracker.log_metrics({"f1": float(f), "threshold": float(opt_thresh)})
    print("\n" + "="*30 + "\nREPORT\n" + classification_report(test_bin, final_preds, target_names=['Normal', 'Anomaly']))