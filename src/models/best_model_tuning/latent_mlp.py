import torch.nn as nn

class LatentMLP(nn.Module):
    """A dynamic Multi-Layer Perceptron for classifying 1D Latent Embeddings"""
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Dynamically build hidden layers based on the config list
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        # Final binary classification layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
        
    def train_step(self, features, labels, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        preds = self.forward(features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        return loss.item()
