import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MyResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 100)
        for i in self.model.parameters():
            i.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)