import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def save_training_curve(train_losses, val_losses, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_f1_bar_chart(y_true, y_pred, labels, out_path):
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, f1_per_class)
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Class")
    plt.xticks(rotation=20)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_predictions_csv(y_true, y_prob, y_pred, labels, out_path):
    rows = []
    for i in range(len(y_true)):
        row = {"sample_index": i}
        for j, label in enumerate(labels):
            row[f"{label}_true"] = int(y_true[i][j])
            row[f"{label}_prob"] = float(y_prob[i][j])
            row[f"{label}_pred"] = int(y_pred[i][j])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)