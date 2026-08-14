"""Global project configuration: paths, hyperparameters."""
from pathlib import Path

# ===== Paths =====
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Original GTSRB dataset (Kaggle format): can be zip or extracted directory
GTSRB_ZIP_PATH = ROOT_DIR / "GTSRB.zip"
GTSRB_RAW_DIR = DATA_DIR / "gtsrb_raw"        # where zip is extracted
CLASSES_FILE = DATA_DIR / "classes.txt"
CLASSES_EN_FILE = DATA_DIR / "classes_en.txt"
CLASSES_VIE_FILE = DATA_DIR / "classes_vie.txt"

# Cropped dataset (classification format)
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_TRAIN_DIR = PROCESSED_DIR / "train"
PROCESSED_VAL_DIR = PROCESSED_DIR / "val"
PROCESSED_TEST_DIR = PROCESSED_DIR / "test"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ===== Data =====
IMG_SIZE = 48                # Resize dimension (HxW) — standard for GTSRB
IMG_CHANNELS = 3             # 3 = RGB, 1 = grayscale
SEED = 42

# ===== Training =====
BATCH_SIZE = 128
EPOCHS = 60
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 12
LR_REDUCE_PATIENCE = 3

# ===== Model =====
# GTSRB (43 German traffic sign classes — default)
MODEL_NAME = "custom_cnn_v1"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.keras"
LABELS_PATH = MODELS_DIR / "labels.json"

# VN (Vietnam traffic signs — fine-tuned from GTSRB)
MODEL_NAME_VN = "custom_cnn_vn_v1"
MODEL_PATH_VN = MODELS_DIR / f"{MODEL_NAME_VN}.keras"
LABELS_PATH_VN = MODELS_DIR / "labels_vn.json"
