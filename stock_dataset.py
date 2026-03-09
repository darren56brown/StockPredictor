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
        self.all_means = [] # Store full vector of means
        self.all_stds = []  # Store full vector of stds
        self.total_windows = 0
        
        # We find 'time_of_day' index to exclude it from the model's output/loss later
        self.time_col_idx = None 

        for f in all_files:
            # Corrected ticker extraction: get name before the .parquet
            ticker = os.path.basename(f).split('.')[0].upper()
            if whitelist and ticker not in [w.upper() for w in whitelist]:
                continue

            try:
                df = pd.read_parquet(f)
                
                if self.time_col_idx is None:
                    if 'time_of_day' in df.columns:
                        self.time_col_idx = list(df.columns).index('time_of_day')
                    else:
                        # Fallback/Error if column missing
                        raise ValueError(f"File {f} missing 'time_of_day' column.")

                # Calculate stats for ALL columns
                m = df.mean().values.astype(np.float32)
                s = df.std().values.astype(np.float32)
                # Prevent division by zero for static columns
                s[s == 0] = 1e-9 

                # Apply Z-score normalization
                df_norm = (df - m) / s
                tensor_data = torch.tensor(df_norm.values, dtype=torch.float32)
                
                num_windows = len(tensor_data) - self.seq_length
                
                if num_windows > 0:
                    self.file_offsets.append(self.total_windows)
                    self.all_tensors.append(tensor_data)
                    self.all_means.append(torch.tensor(m))
                    self.all_stds.append(torch.tensor(s))
                    self.total_windows += num_windows
                else:
                    print(f"Skipping {ticker}: Need at least {self.seq_length + 1} rows.")
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")
        
        self.file_offsets = np.array(self.file_offsets)
        print(f"Dataset initialized: {len(self.all_tensors)} tickers, {self.total_windows} windows.")

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # Find which ticker file the global index belongs to
        file_idx = np.searchsorted(self.file_offsets, idx, side='right') - 1
        local_start = idx - self.file_offsets[file_idx]
        
        data = self.all_tensors[file_idx]
        
        # X: 1500 rows of history
        x = data[local_start : local_start + self.seq_length]
        # Y: The 1501st row (all features for prediction)
        y = data[local_start + self.seq_length]
        
        # Return x.T (for Conv1d), the full target, and stats for de-normalization
        return x.T, y, self.all_means[file_idx], self.all_stds[file_idx]
