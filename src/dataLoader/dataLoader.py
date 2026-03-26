import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_anomaly_dataloaders(
    train_path, 
    test_path, 
    n_train_normal=5000, 
    n_test_normal=250, 
    n_test_anomaly=30, 
    img_size=224, 
    batch_size=32,
    num_workers=16  # <--- Parameter added here
):
    """
    Creates DataLoaders for Anomaly Detection.
    - Train: Only 'NORMAL' class images.
    - Test: A mix of 'NORMAL' and 'Anomaly' (everything else) images.
    """
    
    # Standard normalization for pretrained models
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3), # Ensure 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. Load Datasets
    full_train_ds = datasets.ImageFolder(root=train_path, transform=transform)
    full_test_ds = datasets.ImageFolder(root=test_path, transform=transform)
    
    normal_idx = full_train_ds.class_to_idx['NORMAL']

    # 2. Filter Indices
    def get_indices(dataset, is_normal=True):
        if is_normal:
            return [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == normal_idx]
        else:
            return [i for i, (_, lbl) in enumerate(dataset.samples) if lbl != normal_idx]

    train_norm_idx = get_indices(full_train_ds, is_normal=True)
    test_norm_idx = get_indices(full_test_ds, is_normal=True)
    test_anom_idx = get_indices(full_test_ds, is_normal=False)

    # 3. Create Subsets with requested sizes
    train_subset = Subset(full_train_ds, train_norm_idx[:n_train_normal])
    
    test_combined_idx = test_norm_idx[:n_test_normal] + test_anom_idx[:n_test_anomaly]
    test_subset = Subset(full_test_ds, test_combined_idx)

    # 4. Wrap in Loaders (UPDATED FOR GPU)
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, # <--- Passed into the loader here
        pin_memory=True,         # <--- Added for fast GPU transfer
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_subset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, # <--- Passed into the loader here
        pin_memory=True,         # <--- Added for fast GPU transfer
        persistent_workers=True
    )

    print(f"--- Data Summary ---")
    print(f"Training on: {len(train_subset)} Normal images")
    print(f"Testing on:  {n_test_normal} Normal + {min(len(test_anom_idx), n_test_anomaly)} Anomaly images")
    
    return train_loader, test_loader, normal_idx