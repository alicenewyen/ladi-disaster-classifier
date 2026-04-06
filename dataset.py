from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

from config import TARGET_LABELS, IMG_SIZE


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    return train_tf, eval_tf


class LADIDataset(Dataset):
    def __init__(self, split, transform=None):
        self.data = split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            [float(item[l]) for l in TARGET_LABELS],
            dtype=torch.float32
        )

        return image, label


def load_data():
    ds = load_dataset("MITLL/LADI-v2-dataset")
    return ds["train"], ds["validation"], ds["test"]