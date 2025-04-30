"""
Training Script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.reading_order.model import ReadingOrderModel
from data.reading_order_dataset import ReadingOrderDataset

def train():
    # Hyperparameters
    batch_size = 64
    lr = 1e-4
    num_epochs = 20

    # Load dataset (.json format X: [x1,y1,w1,h1,x2,y2,w2,h2], Y: 0/1)
    train_dataset = ReadingOrderDataset("data/train.json")
    val_dataset = ReadingOrderDataset("data/val.json")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReadingOrderModel().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for coords, semantics, labels in train_loader:
            coords, semantics, labels = coords.to(device), semantics.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(coords, semantics)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for coords, semantics, labels in val_loader:
                coords, semantics, labels = coords.to(device), semantics.to(device), labels.to(device)
                outputs = model(coords, semantics)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Save best model only (one single checkpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/reading_order/best_model.pt")
            print(f"✅ Saved new best model at epoch {epoch+1}")

if __name__ == "__main__":
    """
    Expected Dataset Format:
    -------------------------
    The training and validation JSON files must contain:
    {
        "X": [[x1, y1, w1, h1, x2, y2, w2, h2], ...],  # shape: (N, 8)
        "semantic_embedding_1": [[...], ...],         # shape: (N, D)
        "semantic_embedding_2": [[...], ...],         # shape: (N, D)
        "Y": [0 or 1, ...]                           # shape: (N,)
    }
    Only one model ('best_model.pt') will be saved during training.
    """
    train()