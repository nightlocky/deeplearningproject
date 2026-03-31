import os
import torch

# --- Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # /workspace
TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'OCT', 'train')
TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'OCT', 'test')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'graphs')
# --- Hardware ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 16

# --- Data Parameters ---
IMG_SIZE = 224
BATCH_SIZE = 128

# --- Dataset Splits ---
N_TRAIN_NORMAL = 40000 
N_TEST_NORMAL = 5000
N_TEST_ANOMALY = 750

# Test Run
TEST_N_TRAIN_NORMAL = 1000 
TEST_N_TEST_NORMAL = 500
TEST_N_TEST_ANOMALY = 50
# --- Reproducibility ---
RANDOM_SEED = 42