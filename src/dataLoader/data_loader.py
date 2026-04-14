import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from src import config

def dataloader(
    train_path, 
    test_path, 
    n_train_normal=config.N_TRAIN_NORMAL, 
    n_test_normal=config.N_TEST_NORMAL, 
    n_test_anomaly_per_class=config.N_TEST_ANOMALY_PER_CLASS,
    img_size=config.IMG_SIZE, 
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS 
):
    """
    Creates DataLoaders for Anomaly Detection.
    - Train: Only 'NORMAL' class images.
    - Test: A mix of 'NORMAL' and an equal, stratified amount of Anomaly images.
    """
    
    # Standard normalization for pretrained models
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1), # Ensure 3 channels
        transforms.ToTensor(),
    ])

    # 1. Load Datasets
    full_train_ds = datasets.ImageFolder(root=train_path, transform=transform)
    full_test_ds = datasets.ImageFolder(root=test_path, transform=transform)
    
    normal_idx = full_train_ds.class_to_idx['NORMAL']

    # 2. Filter Indices (UPDATED FOR STRATIFIED SAMPLING)
    # Get all training normal indices
    train_norm_idx = [i for i, (_, lbl) in enumerate(full_train_ds.samples) if lbl == normal_idx]
    
    # Group test indices by class
    test_class_indices = {lbl: [] for lbl in full_test_ds.class_to_idx.values()}
    for i, (_, lbl) in enumerate(full_test_ds.samples):
        test_class_indices[lbl].append(i)

    # Extract test normal indices
    test_norm_idx = test_class_indices[normal_idx][:n_test_normal]

    # Extract exactly N anomaly images from EACH disease class
    test_anom_idx = []
    anomaly_count = 0
    for lbl, indices in test_class_indices.items():
        if lbl != normal_idx:
            # Grab the specific amount for this specific disease
            selected_indices = indices[:n_test_anomaly_per_class]
            test_anom_idx.extend(selected_indices)
            anomaly_count += len(selected_indices)

    # 3. Create Subsets
    train_subset = Subset(full_train_ds, train_norm_idx[:n_train_normal])
    test_combined_idx = test_norm_idx + test_anom_idx
    test_subset = Subset(full_test_ds, test_combined_idx)

    # 4. Wrap in Loaders 
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False, 
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_subset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"--- Data Summary ---")
    print(f"Training on: {len(train_subset)} Normal images")
    print(f"Testing on:  {len(test_norm_idx)} Normal + {anomaly_count} Anomaly images ({len(test_class_indices)-1} classes * {n_test_anomaly_per_class})")
    
    return train_loader, test_loader, normal_idx
