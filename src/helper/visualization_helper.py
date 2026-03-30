import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import torch

from .. import config

def _prepare_save_path(save_path, model_name=None):
    """
    Ensures the model-specific subdirectories exist inside the universal graphs folder.
    """
    if not save_path:
        return None
    
    # Use the universal directory from config
    if model_name:
        target_dir = os.path.join(config.GRAPHS_DIR, model_name)
    else:
        target_dir = config.GRAPHS_DIR
        
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, save_path)

def plot_loss(losses, save_path=None, model_name=None):
    """Plots the training loss over epochs."""
    epochs = range(1, len(losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Linear scale
    ax1.plot(epochs, losses, 'b-o', label='Training Loss')
    ax1.set_title('Training Loss (Linear Scale)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Logarithmic scale
    ax2.plot(epochs, losses, 'r-o', label='Training Loss (Log)')
    ax2.set_yscale('log')
    ax2.set_title('Training Loss (Logarithmic Scale)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss (Log)')
    ax2.grid(True, which="both", ls="-")
    ax2.legend()
    
    plt.tight_layout()
    
    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
        
    plt.close(fig) # MEMORY FIX: Closes the figure to free up RAM
    return final_path

def plot_error_distribution(train_errors, test_normal_errors, test_anomaly_errors, threshold, save_path=None, model_name=None):
    """Plots a clean distribution of reconstruction errors using Seaborn."""
    plt.figure(figsize=(10, 6))
    
    # Optional: Applies a modern, clean background grid automatically
    sns.set_theme(style="whitegrid") 
    
    sns.histplot(train_errors, bins=50, color='green', label='Train (Normal)', stat='density', alpha=0.4, edgecolor=None)
    sns.histplot(test_normal_errors, bins=50, color='blue', label='Test (Normal)', stat='density', alpha=0.4, edgecolor=None)
    sns.histplot(test_anomaly_errors, bins=50, color='red', label='Test (Anomaly)', stat='density', alpha=0.4, edgecolor=None)
    
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.5f})')
    
    tp = np.sum(test_anomaly_errors > threshold)
    fn = np.sum(test_anomaly_errors <= threshold)
    tn = np.sum(test_normal_errors <= threshold)
    fp = np.sum(test_normal_errors > threshold)
    
    plt.title('Score/Error Distribution & Detection Performance', fontsize=14, pad=15)
    plt.xlabel('Reconstruction Error / Score')
    plt.ylabel('Density')
    plt.legend()
    

    stats_text = f"Detection Summary:\nTP: {tp} | FN: {fn}\nTN: {tn} | FP: {fp}"
    plt.gca().text(0.95, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='center', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    
    # 5. Save and Close
    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
        
    plt.close() # Memory fix
    return final_path

def plot_confusion_matrix(cm, target_names, precision, recall, f1, save_path=None, model_name=None):
    """Plots a clean confusion matrix using Seaborn and pre-calculated metrics."""
    plt.figure(figsize=(7, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=target_names, yticklabels=target_names)

    plt.title(f"Confusion Matrix\nPrecision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
        
    plt.close() 
    return final_path


def plot_anomaly_comparison(orig_imgs, recon_imgs, scores, indices, title, filename, model_name=None):
    """
    Plots a 3-column comparison: Original, Reconstruction, and Error Heatmap.
    """
    num_samples = len(indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    fig.suptitle(title, fontsize=16)

    for i, idx in enumerate(indices):
        orig = orig_imgs[idx].transpose(1, 2, 0) if orig_imgs[idx].ndim == 3 else orig_imgs[idx]
        recon = recon_imgs[idx].transpose(1, 2, 0) if recon_imgs[idx].ndim == 3 else recon_imgs[idx]
        
        # Calculate pixel-wise error heatmap
        diff = np.mean((orig - recon)**2, axis=-1) if orig.ndim == 3 else (orig - recon)**2
        
        # Original
        axes[i, 0].imshow(orig, cmap='gray' if orig.ndim == 2 else None)
        axes[i, 0].set_title(f"Original (Idx: {idx})")
        axes[i, 0].axis('off')
        
        # Reconstruction
        axes[i, 1].imshow(recon, cmap='gray' if recon.ndim == 2 else None)
        axes[i, 1].set_title(f"Reconstruction")
        axes[i, 1].axis('off')
        
        # Heatmap
        sns.heatmap(diff, ax=axes[i, 2], cmap='jet', cbar=False)
        axes[i, 2].set_title(f"Error Score: {scores[idx]:.4f}")
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    final_path = _prepare_save_path(filename, model_name)
    if final_path:
        plt.savefig(final_path)
    plt.close()

def generate_anomaly_analysis(model, feature_tensor, raw_images_tensor, all_errors, y_true, model_name):
    """
    Universal analyzer: Reconstructs features, finds top hits/misses, 
    and saves heatmap comparisons.
    """
    print(f"\n[{model_name}] Generating Heatmap Comparisons...")
    
    # 1. Get Reconstructions
    model.eval()
    with torch.no_grad():
        # Move to device for inference, then back to CPU numpy for plotting
        device = next(model.parameters()).device
        recons = model(feature_tensor.to(device)).cpu().numpy()
    
    raw_images = raw_images_tensor.numpy()
    
    # 2. Identify Indices (Focusing only on actual anomalies: Label == 1)
    anomaly_mask = (y_true == 1)
    anom_scores = all_errors[anomaly_mask]
    real_indices = np.where(anomaly_mask)[0]
    
    if len(real_indices) == 0:
        print("Warning: No anomalies found in test set to analyze.")
        return

    # Top 10 Hits (Highest Error - Model is very confident they are anomalies)
    top_hits_sub_idx = np.argsort(anom_scores)[-10:][::-1]
    top_hits_indices = real_indices[top_hits_sub_idx]
    
    # Top 10 Misses (Lowest Error - The "worst" False Negatives)
    top_miss_sub_idx = np.argsort(anom_scores)[:10]
    top_miss_indices = real_indices[top_miss_sub_idx]

    # 3. Trigger Plotting (Using the plot_anomaly_comparison function created earlier)
    plot_anomaly_comparison(
        raw_images, recons, all_errors, top_hits_indices,
        "Top 10 Correctly Identified Anomalies", "heatmaps_best_hits.png", model_name
    )
    
    plot_anomaly_comparison(
        raw_images, recons, all_errors, top_miss_indices,
        "Top 10 Missed Anomalies (False Negatives)", "heatmaps_worst_misses.png", model_name
    )