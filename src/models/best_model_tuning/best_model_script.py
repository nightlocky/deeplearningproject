import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim

# Custom Imports
from src import config
from src.models.best_model_tuning import config_tune as tune_config
from src.dataLoader.data_loader import dataloader
from src.helper.visualization_helper import (
    plot_loss,
    plot_confusion_matrix,
    plot_error_distribution,
    plot_prediction_sample,
)
from src.helper.mlflow_helper import MLFlowTracker
from src.helper.early_stopping import EarlyStopping 
from src.models.best_model_tuning.latent_mlp import LatentMLP


def _make_safe_name(name):
    return name.replace("/", "_").replace("\\", "_")


def _make_safe_label(name):
    return "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")


def _save_breakdown_table(rows, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_height = max(3.5, 0.6 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Class", "Class Type", "Correct", "Wrong", "Total", "Accuracy"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f5d8a")
        elif rows[row_idx - 1][1] == "Aggregate":
            cell.set_facecolor("#e8f1fb")
        else:
            cell.set_facecolor("#f8fbff" if row_idx % 2 == 0 else "#eef5fb")

    ax.set_title(title, fontsize=14, pad=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _build_prediction_breakdown(test_raw, final_preds, normal_idx, class_names):
    rows = []

    def add_row(class_name, class_type, actual_mask, expected_label):
        total = int(np.sum(actual_mask))
        if total == 0:
            correct = 0
        else:
            correct = int(np.sum(final_preds[actual_mask] == expected_label))
        wrong = total - correct
        accuracy = f"{(correct / total):.3f}" if total else "0.000"
        rows.append([class_name, class_type, correct, wrong, total, accuracy])

    normal_mask = test_raw == normal_idx
    anomaly_mask = test_raw != normal_idx

    add_row(class_names[normal_idx], "Aggregate", normal_mask, expected_label=0)
    add_row("ANOMALY", "Aggregate", anomaly_mask, expected_label=1)

    for class_id, class_name in enumerate(class_names):
        if class_id == normal_idx:
            continue
        add_row(class_name, "Disease", test_raw == class_id, expected_label=1)

    return rows


def _set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# 1. Phase 1: Train Autoencoder
# ---------------------------------------------------------
def train_ae(backbone, loss_alpha, run_name, train_loader, save_graphs=True):
    print(f"\n--- Training Autoencoder: {run_name} ---")
    run_dir = os.path.join(config.TUNING_DIR, tune_config.EXPERIMENT_NAME, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    safe_name = _make_safe_name(run_name)
    model_save_path = os.path.join(run_dir, f"{safe_name}_best_ae.pth")

    ae_model = smp.Unet(
        encoder_name=backbone, encoder_weights="imagenet",
        in_channels=1, classes=1, activation="sigmoid"
    ).to(config.DEVICE)

    optimizer = optim.Adam([
        {'params': ae_model.encoder.parameters(), 'lr': tune_config.ENCODER_LR},
        {'params': list(ae_model.decoder.parameters()) + list(ae_model.segmentation_head.parameters()), 'lr': tune_config.DECODER_LR}
    ])

    early_stopping = EarlyStopping(patience=10, path=model_save_path)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda') 
    l1_criterion = nn.L1Loss() 

    train_losses = []
    
    for epoch in range(tune_config.TUNING_EPOCHS):
        ae_model.train()
        batch_losses = []
        for imgs, _ in tqdm(train_loader, desc=f"AE Epoch {epoch+1}"):
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                recon = ae_model(imgs)
                loss_ssim = 1 - ssim(recon, imgs, data_range=1.0)
                loss_l1 = l1_criterion(recon, imgs)
                loss = (loss_alpha * loss_ssim) + ((1.0 - loss_alpha) * loss_l1)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(loss.item())
            
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        early_stopping(epoch_loss, ae_model)
        if early_stopping.early_stop: break
            
    early_stopping.save_checkpoint() 

    if save_graphs:
        plot_loss(train_losses, os.path.join(run_dir, "ae_training_loss_curve.png"), safe_name)
        
    del ae_model, optimizer
    torch.cuda.empty_cache()
    return model_save_path

# ---------------------------------------------------------
# 2. Phase 2: Feature Extraction
# ---------------------------------------------------------
def extract_features(backbone, model_path, test_loader, normal_idx):
    print("\n--- Extracting Features ---")
    ae_model = smp.Unet(encoder_name=backbone, encoder_weights=None, in_channels=1, classes=1, activation="sigmoid").to(config.DEVICE)
    ae_model.load_state_dict(torch.load(model_path, weights_only=True))
    ae_model.eval()

    latents, maps, folder_ids, origs, recons = [], [], [], [], []
    pool = nn.AdaptiveAvgPool2d((1, 1))
    
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Extracting"):
            imgs_device = imgs.to(config.DEVICE)
            encoder_features = ae_model.encoder(imgs_device)
            latent_vector = pool(encoder_features[-1]).flatten(1)
            recon = ae_model(imgs_device)
            diff = torch.abs(imgs_device - recon)
            latents.append(latent_vector.cpu()); maps.append(diff.cpu()); folder_ids.extend(lbls.numpy())
            origs.append(imgs.cpu()); recons.append(recon.cpu())
            
    all_latents = torch.cat(latents)
    all_binary_labels = np.array([0 if l == normal_idx else 1 for l in folder_ids])

    del ae_model; torch.cuda.empty_cache()
    return train_test_split(
        all_latents,
        torch.cat(maps),
        all_binary_labels,
        np.array(folder_ids),
        torch.cat(origs),
        torch.cat(recons),
        test_size=0.5,
        stratify=all_binary_labels,
        random_state=tune_config.RANDOM_SEED,
    )

# ---------------------------------------------------------
# 3. Phase 3: Train & Evaluate MLP
# ---------------------------------------------------------
def train_and_evaluate_mlp(data_splits, hidden_layers, dropout, run_name, normal_idx, tracker, class_names, save_graphs=True):
    print(f"\n--- Evaluating MLP: {run_name} ---")
    (val_lat, test_lat, val_maps, test_maps, val_bin, test_bin, val_raw, test_raw, val_orig, test_orig, val_recon, test_recon) = data_splits
    
    run_dir = os.path.join(config.TUNING_DIR, tune_config.EXPERIMENT_NAME, run_name)
    os.makedirs(run_dir, exist_ok=True)
    safe_name = _make_safe_name(run_name)
    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # --- Visualization Helper ---
    def save_sample_visualization(orig, recon, diff, true_bin, pred_bin, score, class_id, filename):
        status = "Correct" if true_bin == pred_bin else "Incorrect"
        class_name = class_names[int(class_id)]
        true_name = "Anomaly" if true_bin == 1 else "Normal"
        pred_name = "Anomaly" if pred_bin == 1 else "Normal"
        title = f"Class: {class_name} | True: {true_name} | Pred: {pred_name} ({status})"
        return plot_prediction_sample(
            raw_image=orig,
            model_output=recon,
            score=score,
            save_path=os.path.join(run_dir, "samples", filename),
            model_name=safe_name,
            view_mode="reconstruction",
            error_map=diff,
            title=title,
        )

    # --- MLP Training ---
    mlp = LatentMLP(input_dim=val_lat.shape[1], hidden_layers=hidden_layers, dropout_rate=dropout).to(config.DEVICE)
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    loader = DataLoader(TensorDataset(val_lat, torch.tensor(val_bin).float()), batch_size=64, shuffle=True)
    for _ in range(tune_config.MLP_EPOCHS):
        for x, l in loader: mlp.train_step(x.to(config.DEVICE), l.to(config.DEVICE), optimizer, criterion)

    # --- Predictions ---
    def get_preds(model, features):
        model.eval(); probs = []
        with torch.no_grad():
            for (x,) in DataLoader(TensorDataset(features), batch_size=128): 
                probs.extend(model(x.to(config.DEVICE)).cpu().numpy())
        return np.array(probs)

    v_probs, t_probs = get_preds(mlp, val_lat), get_preds(mlp, test_lat)
    
    # --- Threshold Search (Optimizing for F1) ---
    best_f1, opt_thresh = 0, 0.5
    for t in np.linspace(v_probs.min(), v_probs.max(), 100):
        f1_temp = precision_recall_fscore_support(val_bin, (v_probs > t).astype(int), average='binary', zero_division=0)[2]
        if f1_temp > best_f1: best_f1, opt_thresh = f1_temp, t

    # --- Final Metric Calculations ---
    final_preds = (t_probs > opt_thresh).astype(int)
    
    # Standard P, R, F1
    precision, recall, f1, _ = precision_recall_fscore_support(test_bin, final_preds, average='binary', zero_division=0)
    
    # Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(test_bin, final_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC Metrics (Uses raw probabilities, not labels)
    auc_roc = roc_auc_score(test_bin, t_probs)
    auc_pr = average_precision_score(test_bin, t_probs)

    # --- MLFlow Tracking ---
    tracker.start_run(run_name=run_name)
    try:
        tracker.log_params({
            "mlp_layers": str(hidden_layers), 
            "mlp_dropout": dropout, 
            "latent_dim": val_lat.shape[1],
            "phase": run_name.split('/')[0]
        })
        
        # LOGGING ALL 6 KEY METRICS
        tracker.log_metrics({
            "test_f1": float(f1),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_specificity": float(specificity),
            "test_auc_roc": float(auc_roc),
            "test_auc_pr": float(auc_pr),
            "opt_threshold": float(opt_thresh)
        })

        if save_graphs:
            cm = confusion_matrix(test_bin, final_preds)
            cm_path = plot_confusion_matrix(
                cm,
                ["Normal", "Anomaly"],
                precision,
                recall,
                f1,
                specificity=specificity,
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                save_path=os.path.join(run_dir, "test_confusion_matrix.png"),
                model_name=safe_name,
            )
            tracker.log_artifact(cm_path)

            distribution_path = plot_error_distribution(
                v_probs[val_bin == 0],
                t_probs[test_bin == 0],
                t_probs[test_bin == 1],
                opt_thresh,
                os.path.join(run_dir, "test_anomaly_score_distribution.png"),
                safe_name,
            )
            tracker.log_artifact(distribution_path)

            breakdown_rows = _build_prediction_breakdown(test_raw, final_preds, normal_idx, class_names)
            breakdown_png_path = _save_breakdown_table(
                breakdown_rows,
                os.path.join(run_dir, "prediction_breakdown_by_class.png"),
                "Prediction Breakdown by Class",
            )
            tracker.log_artifact(breakdown_png_path)

            # Misclassification Visualization
            idx_fn = np.where((test_bin == 1) & (final_preds == 0))[0] 
            if len(idx_fn) > 0:
                worst_fn_indices = idx_fn[np.argsort(t_probs[idx_fn])][:5]
                for rank, i in enumerate(worst_fn_indices):
                    img_path = save_sample_visualization(
                        test_orig[i], test_recon[i], test_maps[i], 
                        test_bin[i], final_preds[i], t_probs[i], test_raw[i], 
                        f"false_negative_{_make_safe_label(class_names[int(test_raw[i])])}_{rank+1}.png"
                    )
                    tracker.log_artifact(img_path)

            idx_fp = np.where((test_bin == 0) & (final_preds == 1))[0]
            if len(idx_fp) > 0:
                worst_fp_indices = idx_fp[np.argsort(t_probs[idx_fp])[::-1]][:5]
                for rank, i in enumerate(worst_fp_indices):
                    img_path = save_sample_visualization(
                        test_orig[i],
                        test_recon[i],
                        test_maps[i],
                        test_bin[i],
                        final_preds[i],
                        t_probs[i],
                        test_raw[i],
                        f"false_positive_{_make_safe_label(class_names[int(test_raw[i])])}_{rank+1}.png",
                    )
                    tracker.log_artifact(img_path)
    finally:
        tracker.end_run()

    mlp_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
    del mlp; gc.collect(); torch.cuda.empty_cache()
    return f1, mlp_state

# ---------------------------------------------------------
# Main Control Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    _set_random_seed(tune_config.RANDOM_SEED)
    train_loader, test_loader, normal_idx = dataloader(
        train_path=config.TRAIN_PATH,
        test_path=config.TEST_PATH,
        img_size=config.IMG_SIZE,
        n_train_normal=config.N_TRAIN_NORMAL,
        n_test_normal=config.N_TEST_NORMAL, 
        n_test_anomaly_per_class=config.N_TEST_ANOMALY_PER_CLASS,
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS
    )
    class_names = list(test_loader.dataset.dataset.classes)
    tracker = MLFlowTracker(experiment_name=tune_config.EXPERIMENT_NAME)

    if tune_config.CURRENT_PHASE == 1:
            for backbone in tune_config.BACKBONES:
                run_name = f"backbone/{backbone}"
                ae_path = train_ae(backbone, 1.0, run_name, train_loader)
                splits = extract_features(backbone, ae_path, test_loader, normal_idx)
                
                # Using defaults from tune_config instead of hardcoding [256, 64] and 0.3
                train_and_evaluate_mlp(
                    splits, 
                    tune_config.DEFAULT_MLP_ARCH, 
                    tune_config.DEFAULT_MLP_DROPOUT, 
                    run_name, 
                    normal_idx, 
                    tracker,
                    class_names,
                )

    elif tune_config.CURRENT_PHASE == 2:
        bb = tune_config.BEST_BACKBONE_SO_FAR
        for alpha in tune_config.LOSS_ALPHAS:
            run_name = f"loss_functions/{alpha}"
            ae_path = train_ae(bb, alpha, run_name, train_loader)
            splits = extract_features(bb, ae_path, test_loader, normal_idx)
            
            # Using defaults from tune_config
            train_and_evaluate_mlp(
                splits, 
                tune_config.DEFAULT_MLP_ARCH, 
                tune_config.DEFAULT_MLP_DROPOUT, 
                run_name, 
                normal_idx, 
                tracker,
                class_names,
            )

    elif tune_config.CURRENT_PHASE == 3:
        bb = tune_config.BEST_BACKBONE_SO_FAR
        ae_path = tune_config.BEST_AE_WEIGHTS_PATH  # Calling the path from config!
        
        print(f"\n--- Loading Pre-trained AE from: {ae_path} ---")
        
        best_splits = extract_features(bb, ae_path, test_loader, normal_idx)
        best_f1, best_mlp_state = -1, None
        
        for arch in tune_config.MLP_ARCHITECTURES:
            for drop in tune_config.MLP_DROPOUTS:
                run_name = f"MLP/{'_'.join(map(str, arch))}_drop_{drop}"
                f1, m_state = train_and_evaluate_mlp(
                    best_splits,
                    arch,
                    drop,
                    run_name,
                    normal_idx,
                    tracker,
                    class_names,
                )
                
                if f1 > best_f1: 
                    best_f1, best_mlp_state = f1, m_state

        if best_mlp_state:
            torch.save(best_mlp_state, os.path.join(config.TUNING_DIR, tune_config.EXPERIMENT_NAME, "best_final_mlp.pth"))
            print(f"\n======================================")
            print(f" PHASE 3 COMPLETE | Best Final F1: {best_f1:.4f}")
            print(f"======================================")

        if best_mlp_state:
            torch.save(best_mlp_state, os.path.join(config.TUNING_DIR, tune_config.EXPERIMENT_NAME, "best_final_mlp.pth"))
