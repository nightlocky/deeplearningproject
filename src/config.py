<<<<<<< HEAD
import os

# ---------------------------------------------------------
# Dataset Root Configuration — edit ONE place for all scripts
# ---------------------------------------------------------
# LOCAL / RunPod: leave DATA_ROOT_OVERRIDE as None — path is auto-detected.
# GOOGLE COLAB  : set DATA_ROOT_OVERRIDE to your dataset path, e.g.:
#                   DATA_ROOT_OVERRIDE = "/content/drive/MyDrive/data"

DATA_ROOT_OVERRIDE = None

# Which dataset sub-folder to use ("OCT" or "chest_xray")
DATASET = "OCT"

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = DATA_ROOT_OVERRIDE if DATA_ROOT_OVERRIDE else os.path.join(_project_root, "data")

TRAIN_PATH = os.path.join(DATA_ROOT, DATASET, "train")
TEST_PATH  = os.path.join(DATA_ROOT, DATASET, "test")
=======
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
>>>>>>> origin/edison
