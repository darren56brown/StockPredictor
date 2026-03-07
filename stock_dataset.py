import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class StockDataset(Dataset):
    def __init__(self, file_path, seq_length=1500):
        """
        Args:
            file_path (str): Path to the .parquet file.
            seq_length (int): Number of time steps for the input (default 1500).
        """
        # Load the binary parquet file
        df = pd.read_parquet(file_path)
        
        # Convert all features to a tensor: (Total_Rows, 10 Channels)
        # Note: 'time' is the index in Parquet, so it's not in the values.
        self.data = torch.tensor(df.values, dtype=torch.float32)
        
        # We need seq_length for input + 1 for the target
        self.seq_length = seq_length
        self.total_steps = len(self.data)
        
        # The specific target index: usually 'stock_close' which is index 3 
        # (open:0, high:1, low:2, close:3, vol:4 ...)
        self.target_col_idx = 3 

    def __len__(self):
        # We can only start a window if there are 1501 rows ahead
        return self.total_steps - self.seq_length

    def __getitem__(self, idx):
        # 1. Get the input sequence (1500, 10)
        # We slice from idx to idx + 1500
        x = self.data[idx : idx + self.seq_length]
        
        # 2. Get the target (the 1501st tick 'close' price)
        y = self.data[idx + self.seq_length, self.target_col_idx]
        
        # 3. Reshape x for PyTorch CNN: (Channels, Sequence_Length)
        # Transpose from (1500, 10) -> (10, 1500)
        x = x.permute(1, 0)
        
        return x, y

# --- Example Usage ---
# dataset = StockDataset('Processed/TSLA_PROD.parquet', seq_length=1500)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for batch_idx, (inputs, targets) in enumerate(dataloader):
#     print(f"Input shape: {inputs.shape}")   # torch.Size([32, 10, 1500])
#     print(f"Target shape: {targets.shape}") # torch.Size([32])
#     break

