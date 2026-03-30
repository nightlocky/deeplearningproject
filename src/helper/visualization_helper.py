import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns

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