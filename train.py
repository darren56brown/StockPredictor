import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from stock_dataset import StockDataset
from models import StockCNN
import os

def save_checkpoint(model, optimizer, epoch, path="model_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {path} (Epoch {checkpoint['epoch']})")
    return checkpoint['epoch']

def train():
    # 1. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 2. Load Data
    full_dataset = StockDataset('Processed/TSLA_PROD.parquet')
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 3. Optional: Load existing model
    checkpoint_path = "model_checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    # 4. Training Loop
    model.train()
    for epoch in range(start_epoch, start_epoch + 10):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.6f}")
        save_checkpoint(model, optimizer, epoch+1)

if __name__ == "__main__":
    train()
