import torch
from PIL import Image
import numpy as np
from torchvision import transforms

from config import *
from model import build_model
from utils import get_predictions


def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    model = build_model(len(TARGET_LABELS)).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))

    model.eval()

    with torch.no_grad():
        out = model(img)
        probs, preds = get_predictions(out.cpu().numpy(), THRESHOLD)

    print("\nPrediction:")
    for label, p, pred in zip(TARGET_LABELS, probs[0], preds[0]):
        print(f"{label}: prob={p:.3f}, pred={pred}")


if __name__ == "__main__":
    predict("sample.jpg")