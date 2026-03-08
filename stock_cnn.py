import torch
import torch.nn as nn
import torch.nn.functional as F

class StockCNN(nn.Module):
    def __init__(self, num_channels=10, seq_length=1500):
        super(StockCNN, self).__init__()
        
        # 1D Convolutional Layers
        # Input: (Batch, 10, 1500)
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2) # Reduces seq_length by half
        
        # After two pools, 1500 -> 750 -> 375
        self.flatten_size = 64 * (seq_length // 4) 
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 1) # Predicts a single 'next tick' value

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
