# tune_config.py

EXPERIMENT_NAME = "best_model_tuning"

# --- PHASE CONTROL ---
# Set this to 1, 2, or 3 to run only that specific part of the tuning
CURRENT_PHASE = 1 

# --- WINNERS FROM PREVIOUS PHASES ---
# Update these as you finish each phase
BEST_BACKBONE_SO_FAR = "resnet50"
BEST_ALPHA_SO_FAR = 1.0

# --- STEP 1: Backbones to Test ---
BACKBONES = ["resnet18", "resnet34", "resnet50"]

# --- STEP 2: Loss Functions to Test ---
LOSS_ALPHAS = [1.0, 0.8, 0.5] 

# --- STEP 3: MLP Hyperparameters to Test ---
MLP_ARCHITECTURES = [[512], [256, 64], [512, 128, 32]]
MLP_DROPOUTS = [0.1, 0.3, 0.5]

# --- Tuning Hyperparameters ---
TUNING_EPOCHS = 30           
ENCODER_LR = 1e-4
DECODER_LR = 1e-3
MLP_EPOCHS = 30