import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import glob

class StockDirectoryDataset(Dataset):
    def __init__(self, directory_path, seq_length=1500, whitelist=None):
        """
        Args:
            directory_path (str): Path to processed Parquet files.
            seq_length (int): Length of input sequence.
            whitelist (list): Optional list of tickers (e.g., ['TSLA', 'AAPL']) 
                             to include. If None, include all.
        """
        self.seq_length = seq_length
        all_files = sorted(glob.glob(os.path.join(directory_path, "*.parquet")))
        
        self.all_tensors = [] 
        self.file_offsets = [] 
        self.total_windows = 0
        self.target_col_idx = 3 # 'close'

        for f in all_files:
            ticker = os.path.basename(f).split('_')[0].upper()
            
            # Filter based on whitelist
            if whitelist is not None and ticker not in [s.upper() for s in whitelist]:
                continue

            df = pd.read_parquet(f)
            tensor_data = torch.tensor(df.values, dtype=torch.float32)
            
            windows_in_file = len(tensor_data) - self.seq_length
            if windows_in_file > 0:
                self.file_offsets.append(self.total_windows)
                self.all_tensors.append(tensor_data)
                self.total_windows += windows_in_file
        
        self.file_offsets = np.array(self.file_offsets)
        print(f"Loaded {len(self.all_tensors)} tickers, total windows: {self.total_windows}")

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.file_offsets, idx, side='right') - 1
        local_idx = idx - self.file_offsets[file_idx]
        data = self.all_tensors[file_idx]
        x = data[local_idx : local_idx + self.seq_length]
        y = data[local_idx + self.seq_length, self.target_col_idx]
        return x.permute(1, 0), y
