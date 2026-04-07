import torch
import torch.nn as nn
import torch.optim as optim
class ErrorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze only the last dimension to prevent batch collapse
        return self.net(x).squeeze(-1)
    
    def train_step(self, maps, labels, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        preds = self.forward(maps)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        return loss.item()