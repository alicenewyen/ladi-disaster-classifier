import numpy as np
from sklearn.metrics import classification_report, f1_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_predictions(logits, threshold=0.5):
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)
    return probs, preds


def compute_metrics(y_true, y_pred, labels):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0
    )

    return macro_f1, micro_f1, report