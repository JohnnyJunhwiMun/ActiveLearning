import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import INPUT_DIM, NUM_CLASSES

class LSTMModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = F.relu(out)
        out, _ = self.lstm2(out)
        out = F.relu(out)
        out, _ = self.lstm3(out)
        out = F.relu(out)
        out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def save(self, path):
        """Save the complete model including architecture and weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.lstm1.input_size,
            'num_classes': self.fc3.out_features
        }, path)
    
    def load(self, path, device='cpu'):
        """Load the complete model including architecture and weights"""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval() 