import torch
import numpy as np
from torch.utils.data import DataLoader

from config import *
from dataset import load_data, LADIDataset, get_transforms
from model import build_model
from utils import (
    get_predictions,
    compute_metrics,
    save_f1_bar_chart,
    save_predictions_csv
)


def evaluate():
    _, _, test_split = load_data()

    _, eval_tf = get_transforms()
    test_ds = LADIDataset(test_split, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = build_model(len(TARGET_LABELS)).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)

            out = model(x)
            probs, preds = get_predictions(out.cpu().numpy(), THRESHOLD)

            all_true.append(y.numpy())
            all_pred.append(preds)
            all_prob.append(probs)

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    y_prob = np.vstack(all_prob)

    macro_f1, micro_f1, report = compute_metrics(y_true, y_pred, TARGET_LABELS)

    print("\nClassification Report:\n")
    print(report)
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    save_f1_bar_chart(y_true, y_pred, TARGET_LABELS, F1_SCORES_PATH)
    save_predictions_csv(y_true, y_prob, y_pred, TARGET_LABELS, TEST_PREDICTIONS_CSV)

    print(f"Saved report to {CLASSIFICATION_REPORT_PATH}")
    print(f"Saved F1 chart to {F1_SCORES_PATH}")
    print(f"Saved predictions CSV to {TEST_PREDICTIONS_CSV}")


if __name__ == "__main__":
    evaluate()