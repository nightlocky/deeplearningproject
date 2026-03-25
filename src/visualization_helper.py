import matplotlib.pyplot as plt
import numpy as np
import os

def _prepare_save_path(save_path, model_name=None):
    """
    Ensures the 'graphs' directory and any model-specific subdirectories exist.
    Returns the full path for saving.
    """
    if not save_path:
        return None
    
    # Base directory is 'graphs'
    base_dir = "graphs"
    
    if model_name:
        target_dir = os.path.join(base_dir, model_name)
    else:
        target_dir = base_dir
        
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, save_path)

def plot_loss(losses, save_path=None, model_name=None):
    """
    Plots the training loss over epochs, including a logarithmic scale version.
    """
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
        print(f"Loss plot saved to {final_path}")
    
    return final_path

def plot_reconstruction_comparison(norm_img, recon_n, anom_img, recon_a, save_path=None, model_name=None):
    """
    Plots original vs reconstructed images for both normal and anomaly cases.
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    
    ax[0,0].imshow(norm_img.squeeze(), cmap='gray')
    ax[0,0].set_title("Normal (Original)")
    ax[0,0].axis('off')
    
    ax[0,1].imshow(recon_n.squeeze(), cmap='gray')
    ax[0,1].set_title("Normal (Reconstruction)")
    ax[0,1].axis('off')
    
    ax[1,0].imshow(anom_img.squeeze(), cmap='gray')
    ax[1,0].set_title("Anomaly (Original)")
    ax[1,0].axis('off')
    
    ax[1,1].imshow(recon_a.squeeze(), cmap='gray')
    ax[1,1].set_title("Anomaly (Reconstruction)")
    ax[1,1].axis('off')
    
    plt.tight_layout()
    
    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
        print(f"Reconstruction comparison saved to {final_path}")
        
    return final_path

def plot_error_distribution(train_errors, test_normal_errors, test_anomaly_errors, threshold, save_path=None, model_name=None):
    """
    Plots the distribution of reconstruction errors for normal and anomaly samples.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(train_errors, bins=50, alpha=0.5, label='Train (Normal)', color='green', density=True)
    ax.hist(test_normal_errors, bins=50, alpha=0.5, label='Test (Normal)', color='blue', density=True)
    ax.hist(test_anomaly_errors, bins=50, alpha=0.5, label='Test (Anomaly)', color='red', density=True)
    
    ax.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.5f})')
    
    # Calculate simple stats for the plot
    tp = sum(e > threshold for e in test_anomaly_errors)
    fn = sum(e <= threshold for e in test_anomaly_errors)
    tn = sum(e <= threshold for e in test_normal_errors)
    fp = sum(e > threshold for e in test_normal_errors)
    
    stats_text = (f"Detection Summary:\n"
                  f"True Positives: {tp}\n"
                  f"False Negatives: {fn}\n"
                  f"True Negatives: {tn}\n"
                  f"False Positives: {fp}")
    
    ax.text(0.95, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_title('Score/Error Distribution & Detection Performance')
    ax.set_xlabel('Score / Error')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
        print(f"Error distribution saved to {final_path}")
        
    return final_path

def plot_confusion_matrix_custom(cm, target_names, save_path=None, model_name=None):
    """
    Plots a confusion matrix with Precision, Recall, and F1-score metrics.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=target_names, yticklabels=target_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Metrics calculation from CM: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Add metrics text at the bottom
    metrics_text = f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}"
    plt.figtext(0.5, 0.01, metrics_text, wrap=True, horizontalalignment='center', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Make room for the metrics text
    
    final_path = _prepare_save_path(save_path, model_name)
    if final_path:
        plt.savefig(final_path)
        print(f"Confusion matrix saved to {final_path}")
        
    return final_path
