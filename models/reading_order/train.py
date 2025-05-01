import torch
from torch.utils.data import DataLoader
from data.reading_order_dataset import ReadingOrderDataset
from model import ReadingOrderModel
from loss import weighted_bce_loss
from optimizer import build_optimizer

def train():
    # Hyperparameters
    batch_size = 64
    lr = 1e-4
    num_epochs = 15

    # Load dataset
    dataset_path = "data/processed/processed.jsonl"
    full_dataset = ReadingOrderDataset(dataset_path)
    split = int(0.9 * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [split, len(full_dataset) - split])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReadingOrderModel().to(device)
    pos_weight = 7.5
    optimizer = build_optimizer(model, lr=lr)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            coords_i = batch['box1'].to(device)
            coords_j = batch['box2'].to(device)
            cls_1 = batch['cls1'].to(device)
            cls_2 = batch['cls2'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(coords_i, coords_j, cls_1, cls_2)
            loss = weighted_bce_loss(outputs.squeeze(), labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                coords_i = batch['box1'].to(device)
                coords_j = batch['box2'].to(device)
                cls_1 = batch['cls1'].to(device)
                cls_2 = batch['cls2'].to(device)
                labels = batch['label'].to(device)

                outputs = model(coords_i, coords_j, cls_1, cls_2)
                loss = weighted_bce_loss(outputs.squeeze(), labels, pos_weight=pos_weight)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} — Train Loss: {train_loss/len(train_loader):.4f} — Val Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/reading_order/best_model.pt")
            print(f"Saved best model at epoch {epoch+1}")

if __name__ == "__main__":
    train()