import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Import your custom classes
from stock_dataset import StockDirectoryDataset
from stock_cnn import StockCNN

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
    """Parses the Processed directory to find all unique tickers."""
    files = [f for f in os.listdir(directory) if f.endswith('_PROD.parquet')]
    return [f.split('_')[0].upper() for f in files]

def train():
    # 1. Device Detection (Pi 5 vs Linux GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System: Running on {device}")

    # 2. Define Ticker Split (Whitelisting for Test)
    processed_dir = '../StockData/Processed'
    if not os.path.exists(processed_dir):
        print(f"Error: {processed_dir} directory not found.")
        return

    all_tickers = get_available_tickers(processed_dir)
    
    # --- EDIT YOUR TEST TICKERS HERE ---
    test_tickers = ['TSLA', 'W', 'ORLY'] # Model will NEVER see these during training
    train_tickers = [t for t in all_tickers if t not in test_tickers]
    
    print(f"Training on: {train_tickers}")
    print(f"Testing on:  {test_tickers}")

    # 3. Create Datasets
    train_ds = StockDirectoryDataset(processed_dir, seq_length=1500, whitelist=train_tickers)
    test_ds = StockDirectoryDataset(processed_dir, seq_length=1500, whitelist=test_tickers)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 4. Initialize Model
    model = StockCNN(num_channels=10, seq_length=1500).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 5. Resume from Checkpoint
    checkpoint_path = "model_checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)

    # 6. Training Loop
    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation Step (Out-of-Sample Test)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}] "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

if __name__ == "__main__":
    train()
