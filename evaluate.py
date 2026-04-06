import torch
import numpy as np
from torch.utils.data import DataLoader

from config import *
from dataset import load_data, LADIDataset, get_transforms
from model import build_model
from utils import get_predictions, compute_metrics


def evaluate():
    _, _, test_split = load_data()

    _, eval_tf = get_transforms()
    test_ds = LADIDataset(test_split, eval_tf)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = build_model(len(TARGET_LABELS)).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))

    model.eval()

    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)

            out = model(x)
            probs, preds = get_predictions(out.cpu().numpy(), THRESHOLD)

            all_true.append(y.numpy())
            all_pred.append(preds)

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)

    macro_f1, micro_f1, report = compute_metrics(
        y_true, y_pred, TARGET_LABELS
    )

    print("\nClassification Report:\n")
    print(report)

    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    evaluate()