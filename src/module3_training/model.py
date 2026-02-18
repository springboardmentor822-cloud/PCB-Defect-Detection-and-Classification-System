import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def build_model(num_classes):
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model
