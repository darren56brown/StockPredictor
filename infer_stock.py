import torch
import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta

# --- HACKABLE CONFIGURATION ---
TICKER = "TSLA"
SEQ_LENGTH = 1500
STEPS_TO_PREDICT = 50       # Future steps (predicted)
PREVIOUS_STEPS_TO_ECHO = 50  # Past steps (actual history for anchor)
MODEL_PATH = "model_checkpoint.pth"
PROCESSED_DIR = "./Processed"
RAW_DIR = "../StockData/Raw" 
OUTPUT_CSV = f"{TICKER}_inference.csv"

# Import the FULL model that includes BatchNorm (matches your checkpoint)
from stock_cnn import StockCNN as CNNModel

def get_stats(ticker):
    """Calculate mean/std from Raw CSV to de-normalize accurately."""
    # Find the file recursively in Raw/
    paths = glob.glob(os.path.join(RAW_DIR, "**", f"{ticker}.csv"), recursive=True)
    if not paths:
        # Fallback to checking for ticker-named files if subfolders vary
        paths = glob.glob(os.path.join(RAW_DIR, f"{ticker}.csv"))
    
    if not paths:
        raise FileNotFoundError(f"Could not find raw file for {ticker} in {RAW_DIR}")
    
    df_raw = pd.read_csv(paths[0])
    # Match the 'Close' column name from your original pull scripts
    col = 'Close' if 'Close' in df_raw.columns else 'close'
    return df_raw[col].mean(), df_raw[col].std()

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model (StockCNN matches the BatchNorm keys in your error)
    model = CNNModel(num_channels=11, seq_length=SEQ_LENGTH).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Load Processed Data for the "Seed" window
    parquet_path = os.path.join(PROCESSED_DIR, f"{TICKER}.parquet")
    if not os.path.exists(parquet_path):
        print(f"Error: Processed file {parquet_path} not found.")
        return
        
    df_proc = pd.read_parquet(parquet_path)
    m_price, s_price = get_stats(TICKER)

    # 3. Extract Historical "Anchor" Data
    results = []
    history_df = df_proc.iloc[-PREVIOUS_STEPS_TO_ECHO:].copy()
    for ts, row in history_df.iterrows():
        # De-normalize: (z * std) + mean
        real_price = (row['close'] * s_price) + m_price
        results.append({
            'time': ts,
            'price': round(real_price, 4),
            'type': 'ACTUAL'
        })

    # 4. Prepare Seed Window (Last 1500 steps)
    # Shape: (1, 11, 1500)
    current_window = torch.tensor(df_proc.values[-SEQ_LENGTH:], dtype=torch.float32).T.unsqueeze(0).to(device)
    last_time = df_proc.index[-1]

    # 5. Recursive Inference Loop
    print(f"Projecting {STEPS_TO_PREDICT} steps for {TICKER}...")

    for _ in range(STEPS_TO_PREDICT):
        with torch.no_grad():
            # Predict next normalized Close
            pred_norm = model(current_window).item()
            pred_real = (pred_norm * s_price) + m_price
            
            # Increment Time
            next_time = last_time + timedelta(minutes=5)
            
            # Deterministic Time of Day (Scaled 0-1, then Z-scored approx)
            # 09:30 (570m) to 16:00 (960m)
            minutes = next_time.hour * 60 + next_time.minute
            tod_decimal = (minutes - 570) / (960 - 570)
            # Center it like the training data (approx mean 0.5, std 0.28)
            tod_norm = (tod_decimal - 0.5) / 0.28 

            results.append({
                'time': next_time,
                'price': round(pred_real, 4),
                'type': 'PREDICTED'
            })

            # Update Window for next recursion
            # Clone the last available column (index -1)
            next_col = current_window[:, :, -1:].clone()
            
            # Update specific indices: Close=3, TOD=10 (adjust if your col order differs)
            next_col[0, 3, 0] = pred_norm
            next_col[0, 10, 0] = tod_norm
            
            # Slide window: remove oldest [:, :, 0], append newest [:, :, -1]
            current_window = torch.cat([current_window[:, :, 1:], next_col], dim=2)
            last_time = next_time

    # 6. Save to CSV
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Saved to {OUTPUT_CSV}")
    print(f"Final Prediction: {results[-1]['price']} at {results[-1]['time']}")

if __name__ == "__main__":
    run_inference()
