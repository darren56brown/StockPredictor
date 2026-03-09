import torch
import torch.nn as nn
import torch.nn.functional as F

class StockCNN(nn.Module):
    def __init__(self, num_channels=11, seq_length=1500):
        super(StockCNN, self).__init__()
        
        # Layer 1: Short-term patterns
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Layer 2: Refine patterns
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Flattening: 1500 -> (Pool 1) 750 -> (Pool 2) 375
        self.flatten_size = 64 * (seq_length // 4) 
        
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        # OUTPUT 10: Everything except time_of_day
        self.fc2 = nn.Linear(128, num_channels - 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) 
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) 
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # No activation here so it can output normalized range (-inf to +inf)
        return x
