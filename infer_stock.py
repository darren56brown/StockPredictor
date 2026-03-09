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
RAW_DIR = "./Raw" 
OUTPUT_CSV = f"{TICKER}_inference.csv"

# Import your model class
from stock_cnn import LightStockCNN as CNNModel

def get_stats(ticker):
    """Recalculate stats from Raw to de-normalize accurately."""
    paths = glob.glob(os.path.join(RAW_DIR, "**", f"{ticker}.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"Could not find raw file for {ticker}")
    df_raw = pd.read_csv(paths[0])
    return df_raw['Close'].mean(), df_raw['Close'].std()

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model & Data
    model = CNNModel(num_channels=11, seq_length=SEQ_LENGTH).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    parquet_path = os.path.join(PROCESSED_DIR, f"{TICKER}.parquet")
    df_proc = pd.read_parquet(parquet_path)
    m_price, s_price = get_stats(TICKER)

    # 2. Extract Historical "Anchor" Data
    # We take the last PREVIOUS_STEPS_TO_ECHO rows from the processed file
    history_df = df_proc.iloc[-PREVIOUS_STEPS_TO_ECHO:].copy()
    
    # De-normalize historical 'close' for the CSV
    results = []
    for ts, row in history_df.iterrows():
        real_price = (row['close'] * s_price) + m_price
        results.append({
            'time': ts,
            'price': round(real_price, 4),
            'type': 'ACTUAL'
        })

    # 3. Prepare Seed Window for Model (The last 1500 steps)
    # Shape: (1, 11, 1500)
    current_window = torch.tensor(df_proc.values[-SEQ_LENGTH:], dtype=torch.float32).T.unsqueeze(0).to(device)
    last_time = df_proc.index[-1]

    # 4. Recursive Inference Loop
    print(f"Projecting {STEPS_TO_PREDICT} steps for {TICKER}...")

    for _ in range(STEPS_TO_PREDICT):
        with torch.no_grad():
            # Predict
            pred_norm = model(current_window).item()
            pred_real = (pred_norm * s_price) + m_price
            
            # Increment Time
            next_time = last_time + timedelta(minutes=5)
            
            # Deterministic Time of Day (09:30 to 16:00 EST)
            minutes = next_time.hour * 60 + next_time.minute
            day_start, day_end = 570, 960 
            new_tod_decimal = (minutes - day_start) / (day_end - day_start)
            # Center it (Z-score approx) to keep it in range for the CNN
            new_tod_norm = (new_tod_decimal - 0.5) / 0.28 

            results.append({
                'time': next_time,
                'price': round(pred_real, 4),
                'type': 'PREDICTED'
            })

            # Update Window: Shift left, insert new values
            # We clone the last column and update the specific features
            next_col = current_window[:, :, -1].clone().unsqueeze(2)
            next_col[:, 3, 0] = pred_norm        # update 'close'
            next_col[:, 10, 0] = new_tod_norm   # update 'time_of_day'
            
            current_window = torch.cat([current_window[:, :, 1:], next_col], dim=2)
            last_time = next_time

    # 5. Export
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(final_df)} rows to {OUTPUT_CSV} (History: {PREVIOUS_STEPS_TO_ECHO}, Future: {STEPS_TO_PREDICT})")

if __name__ == "__main__":
    run_inference()
