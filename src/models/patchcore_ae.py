import os
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
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
BATCH_SIZE = 32 # Keep this reasonable, PatchCore generates a LOT of data
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading datasets...")
train_loader, test_loader, normal_idx = get_anomaly_dataloaders(
    train_path=TRAIN_PATH, 
    test_path=TEST_PATH, 
    img_size=IMG_SIZE,
    n_train_normal = 40000, # Reduced for testing speed, increase later
    n_test_normal = 5000,
    n_test_anomaly=750,
    batch_size=BATCH_SIZE
)

# ---------------------------------------------------------
# 2. PatchCore Feature Extractor (ResNet18)
# ---------------------------------------------------------
print(f"Loading ResNet18 Backbone onto {DEVICE}...")
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)
resnet.eval()

# We use forward hooks to grab the localized texture features before they are flattened
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

resnet.layer2.register_forward_hook(get_features('layer2')) # 128 channels
resnet.layer3.register_forward_hook(get_features('layer3')) # 256 channels

def extract_patch_features(imgs):
    """Extracts and formats PatchCore features from an image batch."""
    with torch.no_grad():
        _ = resnet(imgs.to(DEVICE))
        f2 = features['layer2']
        f3 = features['layer3']
        
        # Resize layer2 feature map to match layer3's spatial dimensions
        f2 = F.interpolate(f2, size=f3.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine the features (128 + 256 = 384 dimensions per patch)
        f_cat = torch.cat([f2, f3], dim=1) 
        
        # PatchCore Average Pooling (blends neighborhood features smoothly)
        f_pool = F.avg_pool2d(f_cat, kernel_size=3, stride=1, padding=1)
        
        B, C, H, W = f_pool.shape
        # Flatten spatial dimensions: Instead of 1 image vector, we get H*W patch vectors
        patches = f_pool.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return patches # Shape: [Batch_Size, Num_Patches_Per_Image, 384]

# ---------------------------------------------------------
# 3. The Patch Autoencoder
# ---------------------------------------------------------
# Takes a 384-D patch feature, compresses it, and rebuilds it.
class PatchAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 64) # The Patch Bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 384)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

ae_model = PatchAutoencoder().to(DEVICE)
criterion = nn.MSELoss(reduction='none') # Notice 'none': we want error per patch
optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

# ---------------------------------------------------------
# 4. Training Loop (On-the-fly extraction to save RAM)
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="PatchCore_Autoencoder")
EPOCHS = 20 # PatchCore trains faster because 1 image = ~196 patches

with tracker as run:
    tracker.log_params({
        "backbone": "resnet18_layer2_layer3",
        "ae_bottleneck": 64,
        "epochs": EPOCHS,
        "strategy": "PatchCore_AE_MaxPatchError"
    })

    print("\nTraining Patch Autoencoder...")
    train_losses = []
    
    for epoch in range(EPOCHS):
        ae_model.train()
        batch_losses = []
        
        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # 1. Extract patches from ResNet
            patches = extract_patch_features(imgs) # [B, 196, 384]
            
            # Flatten all patches into a giant list for the AE
            patches_flat = patches.reshape(-1, 384) 
            
            # 2. Train AE on patches
            recon = ae_model(patches_flat)
            loss = criterion(recon, patches_flat).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            
        avg_loss = np.mean(batch_losses)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

    # ---------------------------------------------------------
    # 5. Evaluation & Image-Level Scoring
    # ---------------------------------------------------------
    ae_model.eval()
    print("\nExtracting test scores...")
    
    def get_image_anomaly_scores(loader):
        image_scores = []
        labels = []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc="Scoring"):
                patches = extract_patch_features(imgs) # [B, 196, 384]
                B, num_patches, dims = patches.shape
                
                recon = ae_model(patches.reshape(-1, dims))
                
                # Calculate MSE per patch
                patch_errors = torch.mean((patches.reshape(-1, dims) - recon)**2, dim=1)
                
                # Reshape back to [Batch, Num_Patches]
                patch_errors = patch_errors.view(B, num_patches)
                
                # THE PATCHCORE SECRET: An image's anomaly score is the max error of its worst patch
                img_max_errors = torch.max(patch_errors, dim=1)[0].cpu().numpy()
                
                image_scores.extend(img_max_errors)
                labels.extend(lbls.numpy())
        return np.array(image_scores), np.array(labels)

    # Score the training set to establish a baseline
    train_scores, _ = get_image_anomaly_scores(train_loader)
    all_test_scores, test_labels_raw = get_image_anomaly_scores(test_loader)

    y_true_all = np.array([0 if l == normal_idx else 1 for l in test_labels_raw])

    # ---------------------------------------------------------
    # 6. F1-Optimization on Validation Split
    # ---------------------------------------------------------
    val_errs, test_errs, val_lbls, test_lbls = train_test_split(
        all_test_scores, y_true_all, test_size=0.5, stratify=y_true_all, random_state=42
    )

    print("\nRunning F1-Maximization on Validation Set...")
    best_thresh, best_f1 = 0, 0
    thresholds_to_test = np.linspace(val_errs.min(), val_errs.max(), 1000)
    
    for t in thresholds_to_test:
        temp_preds = (val_errs > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(val_lbls, temp_preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    tracker.log_metric("optimal_threshold", best_thresh)
    print(f"Optimal Threshold Found: {best_thresh:.6f} (Validation F1: {best_f1:.4f})")

    # ---------------------------------------------------------
    # 7. Final Test Evaluation
    # ---------------------------------------------------------
    test_preds = [1 if e > best_thresh else 0 for e in test_errs]
    precision, recall, final_f1, _ = precision_recall_fscore_support(test_lbls, test_preds, average='binary')
    tracker.log_metrics({"test_precision": precision, "test_recall": recall, "test_f1": final_f1})

    print("\n" + "="*30)
    print("FINAL PERFORMANCE (PATCHCORE + AUTOENCODER)")
    target_names = ['Healthy (Normal)', 'Pathology (Anomaly)']
    cm = confusion_matrix(test_lbls, test_preds)
    print(classification_report(test_lbls, test_preds, target_names=target_names))

    # Visualizations
    MODEL_NAME = "patchcore_ae"
    plot_loss(train_losses, "patchcore_loss.png", MODEL_NAME)
    plot_confusion_matrix_custom(cm, target_names, "patchcore_cm.png", MODEL_NAME)
    plot_error_distribution(
        train_scores, 
        test_errs[np.array(test_lbls) == 0], 
        test_errs[np.array(test_lbls) == 1], 
        best_thresh, "patchcore_dist.png", MODEL_NAME
    )
    plt.show()