import torch
import torch.nn as nn
import torch.nn.functional as F

class StockCNN(nn.Module):
    # Updated default num_channels to 11 to match your new CSV structure
    def __init__(self, num_channels=11, seq_length=1500):
        super(StockCNN, self).__init__()
        
        # Layer 1: Short-term patterns
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Layer 2: Refine patterns
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Flattening 1500 -> (Pool 1) 750 -> (Pool 2) 375
        # 64 channels * 375 length = 24,000
        self.flatten_size = 64 * (seq_length // 4) 
        
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Input shape: (Batch, 11, 1500)
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # -> (B, 32, 750)
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # -> (B, 64, 375)
        
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LightStockCNN(nn.Module):
    # Updated default num_channels to 11
    def __init__(self, num_channels=11, seq_length=1500):
        super(LightStockCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.flatten_size = 64 * (seq_length // 4)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
