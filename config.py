from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

for p in [OUTPUT_DIR, CHECKPOINT_DIR, REPORTS_DIR, PLOTS_DIR, PREDICTIONS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_LABELS = [
    "buildings_any",
    "flooding_any",
    "debris_any"
]

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
THRESHOLD = 0.5

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
TEST_PREDICTIONS_CSV = REPORTS_DIR / "test_predictions.csv"
TRAINING_CURVE_PATH = PLOTS_DIR / "training_curve.png"
F1_SCORES_PATH = PLOTS_DIR / "f1_scores.png"
SAMPLE_PRED_PATH = PLOTS_DIR / "sample_prediction.png"