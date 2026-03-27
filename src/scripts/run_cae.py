import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataLoader.dataLoader import get_anomaly_dataloaders
from utils.training import train_model
from utils.inference import get_reconstruction_errors
from utils.evaluation import evaluate_model
from visualization_helper import plot_loss, plot_error_distribution, plot_confusion_matrix_custom
from mlflow_helper import MLFlowTracker
from models.cae import ConvAutoencoder

# ---------------------------------------------------------
# 1. Config
# ---------------------------------------------------------
CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "epochs": 20,
    "lr": 1e-3,
    "n_train_normal": 10000,
    "n_test_normal": 250,
    "n_test_anomaly": 750,
    "model_name": "cae"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------
# 2. Data
# ---------------------------------------------------------
train_loader, test_loader, normal_idx = get_anomaly_dataloaders(
    train_path=os.path.join(project_root, 'data', 'OCT', 'train'),
    test_path=os.path.join(project_root, 'data', 'OCT', 'test'),
    img_size=CONFIG["img_size"],
    n_train_normal=CONFIG["n_train_normal"],
    n_test_normal=CONFIG["n_test_normal"],
    n_test_anomaly=CONFIG["n_test_anomaly"],
    batch_size=CONFIG["batch_size"]
)

# ---------------------------------------------------------
# 3. Model
# ---------------------------------------------------------
model = ConvAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

# ---------------------------------------------------------
# 4. Training + Tracking
# ---------------------------------------------------------
tracker = MLFlowTracker(experiment_name="CAE_From_Scratch")

with tracker:
    tracker.log_params(CONFIG)

    train_losses = train_model(
        model,
        train_loader,
        optimizer,
        criterion,
        DEVICE,
        CONFIG["epochs"],
        tracker
    )

    # ---------------------------------------------------------
    # 5. Inference
    # ---------------------------------------------------------
    train_errors, _ = get_reconstruction_errors(model, train_loader, DEVICE)
    test_errors, test_labels_raw = get_reconstruction_errors(model, test_loader, DEVICE)

    test_labels = (test_labels_raw != normal_idx).astype(int)

    # split validation/test
    from sklearn.model_selection import train_test_split
    val_errs, test_errs, val_lbls, test_lbls = train_test_split(
        test_errors,
        test_labels,
        test_size=0.5,
        stratify=test_labels,
        random_state=42
    )

    # ---------------------------------------------------------
    # 6. Evaluation
    # ---------------------------------------------------------
    results = evaluate_model(val_errs, val_lbls, test_errs, test_lbls)

    tracker.log_metrics(results["metrics"])
    tracker.log_metric("threshold", results["threshold"])

    print("\nFinal Metrics:", results["metrics"])

    # ---------------------------------------------------------
    # 7. Visualization
    # ---------------------------------------------------------
    cm = results["confusion_matrix"]
    preds = results["preds"]

    plot_loss(train_losses, "cae_loss.png", "cae")
    plot_confusion_matrix_custom(cm, ['Normal', 'Anomaly'], "cae_cm.png", "cae")

    plot_error_distribution(
        train_errors,
        test_errs[test_lbls == 0],
        test_errs[test_lbls == 1],
        results["threshold"],
        "cae_dist.png",
        "cae"
    )