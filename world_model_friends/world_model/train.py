import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from world_model_friends.config import get_config
from world_model_friends.world_model.dataset import WorldModelDataset, collate_fn
from world_model_friends.world_model.jepa import JEPAPredictor


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
    # config
    num_heads = get_config("train", "num_heads")  # Make sure emb_dim % num_heads == 0
    num_speakers = len(get_config("process", "main_characters")) + 1
    emb_dim = get_config("embeddings", "dimension")
    epochs = get_config("train", "epochs")
    best_val_loss = float("inf")
    patience = get_config("train", "patience")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print(f"Using device: {device}")

    # convert the data to the pytorch format
    train_ds = WorldModelDataset(training_df)
    val_ds = WorldModelDataset(validation_df)
    print()
    print("Loaded datasets to pytorch.")

    # define the model
    model = JEPAPredictor(
        num_speakers=num_speakers,
        emb_dim=emb_dim,
        num_heads=num_heads,
        dropout=get_config("train", "dropout"),
    ).to(device)
    print()
    print("Loaded the model.")

    optimizer = optim.Adam(
        model.parameters(),
        lr=get_config("train", "learning_rate"),
        weight_decay=get_config("train", "weight_decay"),
    )
    print()
    print("Defined the optimizer.")

    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=get_config("train", "scheduler_mode"),
        factor=get_config("train", "scheduler_factor"),
        patience=get_config("train", "scheduler_patience"),
    )
    print()
    print("Defined the scheduler.")

    train_loader = DataLoader(
        train_ds,
        batch_size=get_config("train", "batch_size"),
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=get_config("train", "batch_size"),
        shuffle=False,
        collate_fn=collate_fn,
    )
    print()
    print("Defined data loaders.")

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
