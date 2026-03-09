import torch
import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta
import pytz  # Standard library for timezone handling

# --- HACKABLE CONFIGURATION ---
TICKER = "TSLA"
SEQ_LENGTH = 1500
STEPS_TO_PREDICT = 50
PREVIOUS_STEPS_TO_ECHO = 50
MODEL_PATH = "model_checkpoint.pth"
PROCESSED_DIR = "./Processed"
RAW_DIR = "./Raw" 
OUTPUT_CSV = f"{TICKER}_inference.csv"

from stock_cnn import StockCNN as CNNModel

def to_eastern(ts_naive):
    """Converts a naive UTC timestamp to US/Eastern wall-clock time."""
    utc_ts = pytz.utc.localize(ts_naive)
    eastern_ts = utc_ts.astimezone(pytz.timezone("US/Eastern"))
    return eastern_ts.replace(tzinfo=None) # Strip tz for clean CSV output

def get_stats(ticker):
    paths = glob.glob(os.path.join(RAW_DIR, "**", f"{ticker}.csv"), recursive=True)
    if not paths: paths = glob.glob(os.path.join(RAW_DIR, f"{ticker}.csv"))
    if not paths: raise FileNotFoundError(f"No raw file for {ticker}")
    df_raw = pd.read_csv(paths[0])
    col = 'Close' if 'Close' in df_raw.columns else 'close'
    return df_raw[col].mean(), df_raw[col].std()

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(num_channels=11, seq_length=SEQ_LENGTH).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    parquet_path = os.path.join(PROCESSED_DIR, f"{TICKER}.parquet")
    df_proc = pd.read_parquet(parquet_path)
    m_price, s_price = get_stats(TICKER)

    # 1. Process Historical Anchor (Convert UTC -> Eastern)
    results = []
    history_df = df_proc.iloc[-PREVIOUS_STEPS_TO_ECHO:].copy()
    for ts, row in history_df.iterrows():
        real_price = (row['close'] * s_price) + m_price
        results.append({
            'time': to_eastern(ts),
            'price': round(real_price, 4),
            'type': 'ACTUAL'
        })

    # 2. Setup Inference Window
    current_window = torch.tensor(df_proc.values[-SEQ_LENGTH:], dtype=torch.float32).T.unsqueeze(0).to(device)
    last_time_utc = df_proc.index[-1]

    # 3. Projection Loop
    print(f"Projecting {TICKER} (Timezone corrected to Eastern)...")
    for _ in range(STEPS_TO_PREDICT):
        with torch.no_grad():
            pred_norm = model(current_window).item()
            pred_real = (pred_norm * s_price) + m_price
            
            # Increment UTC clock
            last_time_utc += timedelta(minutes=5)
            
            # Localize for the output
            local_time = to_eastern(last_time_utc)
            
            # Deterministic TOD (09:30 to 16:00 Local Time)
            minutes = local_time.hour * 60 + local_time.minute
            tod_decimal = (minutes - 570) / (960 - 570)
            tod_norm = (tod_decimal - 0.5) / 0.28 

            results.append({
                'time': local_time,
                'price': round(pred_real, 4),
                'type': 'PREDICTED'
            })

            # Update Window
            next_col = current_window[:, :, -1:].clone()
            next_col[:, 3, 0] = pred_norm
            next_col[:, 10, 0] = tod_norm
            current_window = torch.cat([current_window[:, :, 1:], next_col], dim=2)

    # 4. Save
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Success! {TICKER} saved to {OUTPUT_CSV}. Start time today: {results[0]['time']}")

if __name__ == "__main__":
    run_inference()
