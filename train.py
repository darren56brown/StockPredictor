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
    # Return the next epoch to start from
    return checkpoint.get('epoch', 0)

def get_available_tickers(directory):
    if not os.path.exists(directory):
        return []
    # Simplified to match your filename.csv structure
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    tickers = [os.path.splitext(f)[0].upper() for f in files]
    return sorted(list(set(tickers)))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System: Running on {device}")

    processed_dir = './Processed'
    all_tickers = get_available_tickers(processed_dir)
    
    # 3. Define Ticker Split
    test_tickers = ['TSLA', 'W', 'ORLY'] 
    train_tickers = [t for t in all_tickers if t not in test_tickers]
    
    print(f"Training on {len(train_tickers)} tickers: {train_tickers}...")
    print(f"Testing on {len(test_tickers)} tickers: {test_tickers}")

    # 4. Load Datasets
    # seq_length=1500 requires files with at least 1501 rows
    train_ds = StockDirectoryDataset(processed_dir, seq_length=1500, whitelist=train_tickers)
    test_ds = StockDirectoryDataset(processed_dir, seq_length=1500, whitelist=test_tickers)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 5. Initialize Model - UPDATED num_channels to 11
    model = CNNModel(num_channels=11, seq_length=1500).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 6. Resume from Checkpoint
    checkpoint_path = "model_checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)

    # 7. Training Loop
    num_epochs = 100
    print(f"Starting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            # Ensure target is (Batch, 1) to match MSELoss output
            targets = targets.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f} ({batch_idx/elapsed:.2f} b/s)", end='\r')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).view(-1, 1)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"\n==> Epoch [{epoch+1}] Complete | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        # Save next epoch number
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

if __name__ == "__main__":
    train()
