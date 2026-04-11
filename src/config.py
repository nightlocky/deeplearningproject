import os
import torch

# --- Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # /workspace
TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'OCT', 'train')
TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'OCT', 'test')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'graphs')
TUNING_DIR = os.path.join(CURRENT_DIR, 'models', 'best_model_tuning')

# --- Hardware ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 16

# --- Data Parameters ---
IMG_SIZE = 224
BATCH_SIZE = 128

# --- Dataset Splits (Main Run) ---
N_TRAIN_NORMAL = 40000 
N_TEST_NORMAL = 10000
N_TEST_ANOMALY_PER_CLASS = 150  

# --- Test Run for debugging ---
TEST_N_TRAIN_NORMAL = 1000 
TEST_N_TEST_NORMAL = 500
TEST_N_TEST_ANOMALY_PER_CLASS = 10   

# --- Reproducibility ---
RANDOM_SEED = 42
