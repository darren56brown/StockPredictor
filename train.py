import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time

# Import your custom classes
from stock_dataset import StockDirectoryDataset
#from stock_cnn import LightStockCNN as CNNModel
from stock_cnn import StockCNN as CNNModel

def save_checkpoint(model, optimizer, epoch, path="model_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"\n---> Checkpoint saved: {path}")

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0)

def get_available_tickers(directory):
    if not os.path.exists(directory):
        return []
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    tickers = [os.path.splitext(f)[0].upper() for f in files]
    return sorted(list(set(tickers)))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System: Running on {device}")

    processed_dir = './Processed'
    all_tickers = get_available_tickers(processed_dir)
    
    # Define Ticker Split
    test_tickers = ['TSLA', 'W', 'ORLY'] 
    train_tickers = [t for t in all_tickers if t not in test_tickers]
    
    print(f"Training on {len(train_tickers)} tickers...")
    print(f"Testing on {len(test_tickers)} tickers...")

    # 1. Load Datasets (Normalizes everything internally)
    train_ds = StockDirectoryDataset(processed_dir, seq_length=1500, whitelist=train_tickers)
    test_ds = StockDirectoryDataset(processed_dir, seq_length=1500, whitelist=test_tickers)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 2. Initialize Model 
    # num_channels=11 (Inputs: 10 market cols + 1 time col)
    # Output: 10 (Predicts everything EXCEPT time)
    model = CNNModel(num_channels=11, seq_length=1500).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 3. Resume from Checkpoint
    checkpoint_path = "model_checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)

    # 4. Identify column mask
    # We want to train on every column EXCEPT 'time_of_day'
    time_idx = train_ds.time_col_idx
    target_mask = [i for i in range(11) if i != time_idx]

    # 5. Training Loop
    num_epochs = 100
    print(f"Starting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (inputs, targets, means, stds) in enumerate(train_loader):
            inputs = inputs.to(device) # (B, 11, 1500)
            targets = targets.to(device) # (B, 11)
            
            # Target is the next candle minus the time column
            targets_filtered = targets[:, target_mask] # (B, 10)
            
            optimizer.zero_grad()
            outputs = model(inputs) # (B, 10)
            
            loss = criterion(outputs, targets_filtered)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}", end='\r')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, means, stds in test_loader:
                inputs = inputs.to(device)
                targets_filtered = targets.to(device)[:, target_mask]
                outputs = model(inputs)
                val_loss += criterion(outputs, targets_filtered).item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"\n==> Epoch [{epoch+1}] Complete | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        # Save Checkpoint
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

if __name__ == "__main__":
    train()
