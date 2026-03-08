import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Import your custom classes
from stock_dataset import StockDirectoryDataset
from stock_cnn import LightStockCNN as CNNModel

def save_checkpoint(model, optimizer, epoch, path="model_checkpoint.pth"):
    """Saves the model weights and optimizer state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"---> Checkpoint saved: {path}")

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """Loads weights and handles the device mapping (GPU to CPU)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"---> Loaded checkpoint from {path} (Restarting at Epoch {checkpoint['epoch']})")
    return checkpoint['epoch']

def get_available_tickers(directory):
    """Finds all tickers based on .parquet filenames, handling multiple separators."""
    if not os.path.exists(directory):
        return []
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    # Extracts 'TSLA' from 'TSLA.parquet' or 'TSLA_anything.parquet'
    tickers = [f.replace('.', '_').split('_')[0].upper() for f in files]
    return sorted(list(set(tickers)))

def train():
    # 1. Device Detection (Pi 5 vs Linux GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System: Running on {device}")

    # 2. Directory and Ticker Validation
    processed_dir = '../StockData/Processed'
    if not os.path.exists(processed_dir):
        print(f"EXIT: Directory '{processed_dir}' not found. Run your processing script first.")
        sys.exit(1)

    all_tickers = get_available_tickers(processed_dir)
    if not all_tickers:
        print(f"EXIT: No .parquet files found in {processed_dir}")
        sys.exit(1)
    
    # 3. Define Ticker Split
    # Whitelist these for testing; all others will be used for training
    test_tickers = ['TSLA', 'W', 'ORLY'] 
    train_tickers = [t for t in all_tickers if t not in test_tickers]
    
    if not train_tickers:
        print("EXIT: No tickers left for training after test split.")
        sys.exit(1)

    print(f"Training on: {train_tickers}")
    print(f"Testing on:  {test_tickers}")

    # 4. Load Datasets
    train_ds = StockDirectoryDataset(processed_dir, seq_length=1500, whitelist=train_tickers)
    test_ds = StockDirectoryDataset(processed_dir, seq_length=1500, whitelist=test_tickers)

    if len(train_ds) == 0:
        print("EXIT: Training dataset contains 0 windows. Check sequence length vs file size.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 5. Initialize Model
    model = CNNModel(num_channels=10, seq_length=1500).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 6. Resume from Checkpoint
    checkpoint_path = "model_checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)

    # 7. Training Loop
    num_epochs = 10
    print(f"Starting training for {num_epochs} additional epochs...")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        import time
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # --- PI PROGRESS FEEDBACK ---
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed
                print(f"Epoch [{epoch+1}] Batch [{batch_idx+1}/{total_batches}] "
                      f"Loss: {loss.item():.6f} ({batches_per_sec:.2f} batch/s)", end='\r')

        # Clear the progress line for the final epoch summary
        print("") 

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        avg_train_loss = running_loss / total_batches
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"==> Epoch [{epoch+1}] Complete | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

if __name__ == "__main__":
    train()
