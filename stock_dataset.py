import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import glob

class StockDirectoryDataset(Dataset):
    def __init__(self, directory_path, seq_length=1500, whitelist=None):
        self.seq_length = seq_length
        all_files = sorted(glob.glob(os.path.join(directory_path, "*.parquet")))
        
        self.all_tensors = [] 
        self.file_offsets = [] 
        self.total_windows = 0
        
        # We find 'close' dynamically to avoid hardcoding index 3
        self.target_col_idx = None 

        for f in all_files:
            ticker = os.path.basename(f).split('.')[0].upper()
            if whitelist and ticker not in [w.upper() for w in whitelist]:
                continue

            try:
                df = pd.read_parquet(f)
                
                # Identify target column index on first successful load
                if self.target_col_idx is None:
                    if 'close' in df.columns:
                        self.target_col_idx = list(df.columns).index('close')
                    else:
                        raise ValueError(f"File {f} missing 'close' column.")

                tensor_data = torch.tensor(df.values, dtype=torch.float32)
                
                # Number of valid windows is (Total Rows - Seq Length)
                # If we have 1501 rows, we have exactly 1 window (0:1500 for X, 1500 for Y)
                num_windows = len(tensor_data) - self.seq_length
                
                if num_windows > 0:
                    # Store the cumulative start index for this file
                    self.file_offsets.append(self.total_windows)
                    self.all_tensors.append(tensor_data)
                    self.total_windows += num_windows
                else:
                    print(f"Skipping {ticker}: Need at least {self.seq_length + 1} rows.")
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")
        
        self.file_offsets = np.array(self.file_offsets)
        print(f"Loaded {len(self.all_tensors)} tickers. Total windows: {self.total_windows}")

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # Find which file contains the window for global 'idx'
        # side='right' and subtracting 1 gives the correct file index
        file_idx = np.searchsorted(self.file_offsets, idx, side='right') - 1
        
        # Calculate start position within that specific file
        local_start = idx - self.file_offsets[file_idx]
        data = self.all_tensors[file_idx]
        
        # X: [local_start : local_start + 1500]
        # Y: [local_start + 1500] (the 1501st element)
        x = data[local_start : local_start + self.seq_length]
        y = data[local_start + self.seq_length, self.target_col_idx]
        
        # Permute to (Channels, Length) for Conv1d: (11, 1500)
        return x.T, y
