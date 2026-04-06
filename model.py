import torch.nn as nn
from torchvision import models


def build_model(num_labels):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, num_labels)

    return model