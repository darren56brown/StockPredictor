import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys

# Import your custom classes
from stock_dataset import StockDataset
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
    """Loads weights and handles the device mapping (GPU to CPU or vice versa)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"---> Loaded checkpoint from {path} (Restarting at Epoch {checkpoint['epoch']})")
    return checkpoint['epoch']

def train():
    # 1. Device Detection (Works on Pi 5 / ARM and Linux / NVIDIA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System: Running on {device}")

    # 2. Initialize Model and move to device
    model = StockCNN(num_channels=10, seq_length=1500).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 3. Load Data
    data_path = '../StockData/Processed/TSLA.parquet'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run process_stock_data.py first.")
        return

    full_dataset = StockDataset(data_path, seq_length=1500)
    
    # Split: 80% Training, 20% Testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 4. Resume from Checkpoint if it exists
    checkpoint_path = "model_checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)

    # 5. Training Loop
    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train() # Set to training mode
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to the same device as the model
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation Step
        model.eval() # Set to evaluation mode
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
        
        # Save after every epoch
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

if __name__ == "__main__":
    train()
