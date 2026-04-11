# tune_config.py

EXPERIMENT_NAME = "best_model_tuning"

# --- Step 1: Backbones to Test ---
# We are actively tuning this, so we leave the full list.
BACKBONES = ["resnet18", "resnet34", "resnet50"]

# --- Step 2: Loss Functions to Test ---
# Formula: Loss = (alpha * SSIM_Loss) + ((1 - alpha) * L1_Loss)
# LOSS_ALPHAS = [1.0, 0.8, 0.5] 
LOSS_ALPHAS = [1.0]  # Baseline: Pure SSIM

# --- Step 3: MLP Hyperparameters to Test ---
# Each list represents the hidden layer dimensions. 
# e.g., [256, 64] = 2 hidden layers. [512] = 1 hidden layer.
# MLP_ARCHITECTURES = [
#     [512],            # Shallow & Wide
#     [256, 64],        # Your Baseline
#     [512, 128, 32]    # Deep
# ]
# MLP_DROPOUTS = [0.1, 0.3, 0.5]

MLP_ARCHITECTURES = [[256, 64]]  # Baseline architecture
MLP_DROPOUTS = [0.3]             # Baseline dropout

# --- Tuning Hyperparameters ---
TUNING_EPOCHS = 30           
ENCODER_LR = 1e-4
DECODER_LR = 1e-3
MLP_EPOCHS = 30