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
