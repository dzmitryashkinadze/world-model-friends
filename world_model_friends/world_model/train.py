import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from world_model_friends.world_model.dataset import WorldModelDataset, collate_fn
from world_model_friends.world_model.model import WorldModel


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        context_identity = batch["context_identity"].to(device)
        context_embedding = batch["context_embedding"].to(device)
        target_identity = batch["target_identity"].to(device)
        target_embedding = batch["target_embedding"].to(device)

        # Forward pass
        preds = model(context_identity, context_embedding, target_identity)

        # Back propagation
        loss = criterion(preds, target_embedding)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            context_identity = batch["context_identity"].to(device)
            context_embedding = batch["context_embedding"].to(device)
            target_identity = batch["target_identity"].to(device)
            target_embedding = batch["target_embedding"].to(device)
            preds = model(context_identity, context_embedding, target_identity)
            loss = criterion(preds, target_embedding)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main(training_df, validation_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # convert the data to the pytorch format
    train_ds = WorldModelDataset(training_df)
    val_ds = WorldModelDataset(validation_df)
    print("Loaded datasets to pytorch.")

    # Let's use a sample from the dataset to get dimensions
    sample_item = train_ds[0]
    num_speakers = sample_item["target_identity"].shape[0]
    emb_dim = sample_item["target_embedding"].shape[0]

    # define the model
    model = WorldModel(num_speakers=num_speakers, emb_dim=emb_dim, dropout=0.2).to(
        device
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    epochs = 50  # Increased epochs to allow for early stopping
    best_val_loss = float("inf")
    patience = 5
    counter = 0
    print("Ready for training")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early Stopping and Model Saving logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  --> New best model saved (Val Loss: {best_val_loss:.6f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
