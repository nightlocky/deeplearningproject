import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ---------------------------------------------------------
# 1. Threshold Optimization
# ---------------------------------------------------------
def find_best_threshold(errors, labels, num_steps=1000):
    """
    Finds the threshold that maximizes F1 score on validation data.

    Args:
        errors (np.array): anomaly scores
        labels (np.array): ground truth (0 = normal, 1 = anomaly)
        num_steps (int): number of thresholds to try

    Returns:
        best_thresh (float)
        best_f1 (float)
    """
    thresholds = np.linspace(errors.min(), errors.max(), num_steps)

    best_thresh = 0
    best_f1 = 0

    for t in thresholds:
        preds = (errors > t).astype(int)

        _, _, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average='binary',
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


# ---------------------------------------------------------
# 2. Compute Metrics
# ---------------------------------------------------------
def compute_metrics(y_true, y_pred, scores=None):
    """
    Computes evaluation metrics.

    Args:
        y_true (np.array)
        y_pred (np.array)
        scores (np.array, optional): raw anomaly scores

    Returns:
        dict of metrics
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='binary',
        zero_division=0
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # Threshold-independent metrics
    if scores is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, scores)
        except:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = average_precision_score(y_true, scores)
        except:
            metrics["pr_auc"] = None

    return metrics


# ---------------------------------------------------------
# 3. Full Evaluation Pipeline
# ---------------------------------------------------------
def evaluate_model(val_errs, val_lbls, test_errs, test_lbls):
    """
    Full evaluation pipeline:
    - Finds best threshold on validation set
    - Applies on test set
    - Computes metrics

    Returns:
        dict containing all results
    """
    # ---- Threshold tuning ----
    best_thresh, val_f1 = find_best_threshold(val_errs, val_lbls)

    # ---- Predictions ----
    test_preds = (test_errs > best_thresh).astype(int)

    # ---- Metrics ----
    metrics = compute_metrics(
        test_lbls,
        test_preds,
        scores=test_errs
    )

    # ---- Confusion Matrix ----
    cm = confusion_matrix(test_lbls, test_preds)

    return {
        "threshold": best_thresh,
        "val_f1": val_f1,
        "metrics": metrics,
        "preds": test_preds,
        "confusion_matrix": cm
    }