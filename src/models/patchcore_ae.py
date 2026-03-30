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
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim

# Ensure paths are correct
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir) # Targets /workspace
sys.path.append(parent_dir)

# Import your custom helpers
from dataLoader.dataLoader import get_anomaly_dataloaders
from visualization_helper import plot_loss, plot_error_distribution, plot_confusion_matrix_custom
from mlflow_helper import MLFlowTracker

# ---------------------------------------------------------
# 1. Configuration & Data Loading
# ---------------------------------------------------------
TRAIN_PATH = os.path.join(project_root, 'data', 'OCT', 'train')
TEST_PATH = os.path.join(project_root, 'data', 'OCT', 'test')

IMG_SIZE = 224
BATCH_SIZE = 128 
NUM_WORKERS = 16 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, normal_idx = get_anomaly_dataloaders(
    train_path=TRAIN_PATH, 
    test_path=TEST_PATH, 
    img_size=IMG_SIZE,
    n_train_normal=40000, 
    n_test_normal=5000,
    n_test_anomaly=750,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# ---------------------------------------------------------
# 2. PatchCore Feature Extractor (ResNet18)
# ---------------------------------------------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)
resnet.eval()

features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

resnet.layer2.register_forward_hook(get_features('layer2')) 
resnet.layer3.register_forward_hook(get_features('layer3')) 

def extract_patch_features(imgs):
    with torch.no_grad():
        _ = resnet(imgs) 
        f2 = features['layer2']
        f3 = features['layer3']
        f2 = F.interpolate(f2, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f_cat = torch.cat([f2, f3], dim=1) 
        f_pool = F.avg_pool2d(f_cat, kernel_size=3, stride=1, padding=1)
        return f_pool # [B, 384, 28, 28]

# ---------------------------------------------------------
# 3. The Patch Autoencoder (Spatial 2D)
# ---------------------------------------------------------
class PatchAutoencoder(nn.Module):
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

ae_model = PatchAutoencoder().to(DEVICE)
ae_model = torch.compile(ae_model)

optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler('cuda')

# ---------------------------------------------------------
# 4. Training Loop (SSIM)
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="PatchCore_Autoencoder")
EPOCHS = 20 

with tracker as run:
    tracker.log_params({"ae_bottleneck": 64, "epochs": EPOCHS, "loss": "SSIM"})
    
    print("\nTraining Patch Autoencoder...")
    train_losses = []
    
    for epoch in range(EPOCHS):
        ae_model.train()
        batch_losses = []
        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                patches = extract_patch_features(imgs)
                recon = ae_model(patches)
                loss = 1 - ssim(torch.clamp(recon,0,1), torch.clamp(patches,0,1), data_range=1.0)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(loss.item())
            
        train_losses.append(np.mean(batch_losses))

    # ---------------------------------------------------------
    # 5. Evaluation & Image-Level Scoring
    # ---------------------------------------------------------
    ae_model.eval()
    def get_image_anomaly_scores(loader):
        image_scores, labels = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc="Scoring"):
                imgs = imgs.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    patches = extract_patch_features(imgs)
                    recon = ae_model(patches)
                    # Score is the max spatial error
                    err = torch.mean((patches - recon)**2, dim=1) 
                    scores = err.view(imgs.shape[0], -1).max(dim=1)[0]
                image_scores.extend(scores.float().cpu().numpy())
                labels.extend(lbls.numpy())
        return np.array(image_scores), np.array(labels)

    train_scores, _ = get_image_anomaly_scores(train_loader)
    all_test_scores, test_labels_raw = get_image_anomaly_scores(test_loader)
    y_true_all = np.array([0 if l == normal_idx else 1 for l in test_labels_raw])

    # ---------------------------------------------------------
    # 6. F1-Optimization & Metrics Calculation
    # ---------------------------------------------------------
    val_errs, test_errs, val_lbls, test_lbls = train_test_split(
        all_test_scores, y_true_all, test_size=0.5, stratify=y_true_all, random_state=42
    )
    
    thresholds = np.linspace(val_errs.min(), val_errs.max(), 1000)
    best_thresh, best_f1 = 0, 0
    for t in thresholds:
        p = (val_errs > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(val_lbls, p, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    test_preds = (test_errs > best_thresh).astype(int)
    # CALCULATE METRICS FOR THE GRAPH
    precision, recall, f1_final, _ = precision_recall_fscore_support(test_lbls, test_preds, average='binary')
    
    # ---------------------------------------------------------
    # 7. Visualization & Artifacts (MATCHING ViT STYLE)
    # ---------------------------------------------------------
    os.chdir(project_root) 
    MODEL_NAME = "patchcore_ae"
    target_names = ['Normal', 'Anomaly']
    cm = confusion_matrix(test_lbls, test_preds)

    # Pass the calculated metrics so they appear in the confusion matrix box
    l_p = plot_loss(train_losses, "patchcore_loss.png", MODEL_NAME)
    cm_p = plot_confusion_matrix_custom(cm, target_names, "patchcore_cm.png", MODEL_NAME)
    d_p = plot_error_distribution(
        train_scores, 
        test_errs[test_lbls == 0], 
        test_errs[test_lbls == 1], 
        best_thresh, "patchcore_dist.png", MODEL_NAME
    )

    # Log to MLflow
    tracker.log_metrics({"precision": precision, "recall": recall, "f1": f1_final})
    tracker.log_artifact(l_p)
    tracker.log_artifact(cm_p)
    tracker.log_artifact(d_p)
    
    print(f"\nFinal F1: {f1_final:.4f}")