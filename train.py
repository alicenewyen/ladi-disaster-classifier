import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import *
from dataset import load_data, LADIDataset, get_transforms
from model import build_model
from utils import get_predictions, compute_metrics, save_training_curve


def train():
    train_split, val_split, _ = load_data()

    train_tf, eval_tf = get_transforms()

    train_ds = LADIDataset(train_split, train_tf)
    val_ds = LADIDataset(val_split, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = build_model(len(TARGET_LABELS)).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        model.train()
        total_train_loss = 0.0

        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        all_true, all_pred = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                out = model(x)
                loss = criterion(out, y)
                total_val_loss += loss.item()

                probs, preds = get_predictions(out.cpu().numpy(), THRESHOLD)
                all_true.append(y.cpu().numpy())
                all_pred.append(preds)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        y_true = np.vstack(all_true)
        y_pred = np.vstack(all_pred)

        macro_f1, micro_f1, _ = compute_metrics(y_true, y_pred, TARGET_LABELS)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Macro F1:   {macro_f1:.4f}")
        print(f"Micro F1:   {micro_f1:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model to {BEST_MODEL_PATH}")

    save_training_curve(train_losses, val_losses, TRAINING_CURVE_PATH)
    print(f"Saved training curve to {TRAINING_CURVE_PATH}")
    print("Training done.")


if __name__ == "__main__":
    train()