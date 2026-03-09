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
        self.all_stats = [] # Stores {'mean': x, 'std': x} for each ticker
        self.total_windows = 0
        self.target_col_idx = None 

        for f in all_files:
            ticker = os.path.basename(f).split('.')[0].upper()
            if whitelist and ticker not in [w.upper() for w in whitelist]:
                continue

            try:
                df = pd.read_parquet(f)
                
                # Identify 'close' index among the features
                if self.target_col_idx is None:
                    if 'close' in df.columns:
                        self.target_col_idx = list(df.columns).index('close')
                    else:
                        raise ValueError(f"File {f} missing 'close' column.")

                # 1. Calculate Stats (Z-Score)
                # We calculate stats for EVERY column so the model sees 
                # normalized volume and price relative to that specific ticker.
                m = df.mean().values
                s = df.std().values
                s[s == 0] = 1e-9 # Prevent division by zero

                # 2. Apply Normalization
                df_norm = (df - m) / s
                
                # 3. Convert to Tensor (Index is NOT included in .values)
                tensor_data = torch.tensor(df_norm.values, dtype=torch.float32)
                
                num_windows = len(tensor_data) - self.seq_length
                
                if num_windows > 0:
                    self.file_offsets.append(self.total_windows)
                    self.all_tensors.append(tensor_data)
                    
                    # Store only the 'close' stats for de-normalization later
                    self.all_stats.append({
                        'mean': m[self.target_col_idx],
                        'std': s[self.target_col_idx]
                    })
                    
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
        file_idx = np.searchsorted(self.file_offsets, idx, side='right') - 1
        local_start = idx - self.file_offsets[file_idx]
        
        data = self.all_tensors[file_idx]
        stats = self.all_stats[file_idx]
        
        x = data[local_start : local_start + self.seq_length]
        y = data[local_start + self.seq_length, self.target_col_idx]
        
        # Return x.T, normalized y, and the scalars needed to recover the real price
        # x.T shape: (11, 1500)
        return x.T, y, stats['mean'], stats['std']
