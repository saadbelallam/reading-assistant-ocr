import torch
import os
import warnings
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.reading_order_dataset import ReadingOrderDataset
from model import ReadingOrderModel
from loss import weighted_bce_loss
from optimizer import build_optimizer

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
warnings.filterwarnings("ignore")  # Suppress Python warnings
torch.set_warn_always(False)  # Suppress PyTorch warnings

def train():
    # Hyperparameters
    batch_size = 32
    lr = 1e-4
    num_epochs = 15
    pos_weight = 1

    # Load dataset
    dataset_path = "data/processed/processed.jsonl"
    full_dataset = ReadingOrderDataset(dataset_path)

    train_split = int(0.7 * len(full_dataset))
    val_split = int(0.2 * len(full_dataset))
    test_split = len(full_dataset) - train_split - val_split

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_split, val_split, test_split])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReadingOrderModel().to(device)
    optimizer = build_optimizer(model, lr=lr)

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            coords_i = batch['box1'].to(device)
            coords_j = batch['box2'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(coords_i, coords_j)
            loss = weighted_bce_loss(outputs.squeeze(), labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                coords_i = batch['box1'].to(device)
                coords_j = batch['box2'].to(device)
                labels = batch['label'].to(device)

                outputs = model(coords_i, coords_j)
                loss = weighted_bce_loss(outputs, labels, pos_weight=pos_weight)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} — Train Loss: {avg_train_loss:.4f} — Val Loss: {avg_val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch

    print(f"Saved best model at epoch {best_epoch+1}")
    torch.save(best_model_state, "models/reading_order/best_model.pt")

    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    test_loss_nomal = 0.0

    with torch.no_grad():
        for batch in test_loader:
            coords_i = batch['box1'].to(device)
            coords_j = batch['box2'].to(device)
            labels = batch['label'].to(device)

            outputs = model(coords_i, coords_j)
            loss = weighted_bce_loss(outputs.squeeze(), labels, pos_weight=pos_weight)
            test_loss += loss.item()

            bce_loss = weighted_bce_loss(outputs.squeeze(), labels)
            test_loss_nomal += bce_loss.item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_loss_nomal = test_loss_nomal / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Loss (Standard BCE): {avg_test_loss_nomal:.4f}")

    # Plots
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == "__main__":
    train()