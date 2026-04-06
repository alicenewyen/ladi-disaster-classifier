import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from config import *
from model import build_model
from utils import get_predictions


def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    original_img = Image.open(image_path).convert("RGB")
    img_tensor = transform(original_img).unsqueeze(0).to(DEVICE)

    model = build_model(len(TARGET_LABELS)).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        out = model(img_tensor)
        probs, preds = get_predictions(out.cpu().numpy(), THRESHOLD)

    print("\nPrediction:")
    pred_lines = []
    for label, p, pred in zip(TARGET_LABELS, probs[0], preds[0]):
        line = f"{label}: prob={p:.3f}, pred={pred}"
        pred_lines.append(line)
        print(line)

    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.axis("off")
    plt.title("\n".join(pred_lines), fontsize=10)
    plt.tight_layout()
    plt.savefig(SAMPLE_PRED_PATH, dpi=300)
    plt.close()

    print(f"\nSaved sample prediction figure to {SAMPLE_PRED_PATH}")


if __name__ == "__main__":
    predict("sample.jpg")