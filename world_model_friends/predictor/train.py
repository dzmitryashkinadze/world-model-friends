"""
Training script for the JEPA (Joint Embedding Predictive Architecture) world model.

This script handles the training loop, validation, and early stopping logic.
It uses a custom dataset and a JEPA-inspired predictor model.
"""

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from world_model_friends.config import get_config
from world_model_friends.predictor.dataset import WorldModelDataset, collate_fn
from world_model_friends.predictor.jepa import JEPAPredictor


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    val_loader: DataLoader,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The JEPA predictor model.
        dataloader (DataLoader): The training data loader.
        optimizer (optim.Optimizer): The optimizer for updating model parameters.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run the training
            on (e.g., 'cuda' or 'cpu').
        val_loader (DataLoader): Validation data loader.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    total_loss = 0
    running_train_loss = 0
    step = 0
    print("Training tqdm")
    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        step += 1
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

        loss_value = loss.item()
        total_loss += loss_value
        running_train_loss += loss_value

        if step % get_config(section="train", key="running_train_loss_steps") == 0:
            _loss = running_train_loss / get_config(
                section="train", key="running_train_loss_steps"
            )
            print(f"Running training loss: {_loss}")
            running_train_loss = 0

        if step < 10_000 and step % 2000 == 0:
            val_loss = validate(
                model=model, dataloader=val_loader, criterion=criterion, device=device
            )
            print(f"Running validation loss: {val_loss:.6f}")
            model.train()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """
    Validates the model on a validation set.

    Args:
        model (nn.Module): The JEPA predictor model.
        dataloader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run the validation on.

    Returns:
        float: The average loss for the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        print("Validation tqdm")
        for batch in tqdm(dataloader):
            context_identity = batch["context_identity"].to(device)
            context_embedding = batch["context_embedding"].to(device)
            target_identity = batch["target_identity"].to(device)
            target_embedding = batch["target_embedding"].to(device)
            preds = model(context_identity, context_embedding, target_identity)
            loss = criterion(preds, target_embedding)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_world_model(
    train_df: pl.DataFrame, val_df: pl.DataFrame, data_path: str
) -> None:
    """
    Main training entry point.

    Args:
        train_df (pl.DataFrame): The training dataset as a Polars DataFrame.
        val_df (pl.DataFrame): The validation dataset as a Polars DataFrame.
        data_path (str): Tre path to the data folder.

    Returns:
        None
    """
    # config
    n_heads = get_config(
        section="train", key="n_heads"
    )  # Make sure emb_dim % n_heads == 0
    n_speakers = len(get_config(section="process", key="main_characters")) + 1
    emb_dim = get_config(section="embedding", key="dimension")
    epochs = get_config(section="train", key="epochs")
    n_layers = get_config(section="train", key="n_layers", default=2)
    best_val_loss = float("inf")
    patience = get_config(section="train", key="patience")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print(f"Using device: {device}")

    # convert the data to the pytorch format
    train_ds = WorldModelDataset(df=train_df)
    val_ds = WorldModelDataset(df=val_df)
    print()
    print("Loaded datasets to pytorch.")

    # define the model
    model = JEPAPredictor(
        n_speakers=n_speakers,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=get_config(section="train", key="dropout"),
    ).to(device)
    print()
    print("Loaded the model.")

    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=get_config(section="train", key="learning_rate"),
        weight_decay=get_config(section="train", key="weight_decay"),
    )
    print()
    print("Defined the optimizer.")

    criterion = nn.HuberLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=get_config(section="train", key="scheduler_mode"),
        factor=get_config(section="train", key="scheduler_factor"),
        patience=get_config(section="train", key="scheduler_patience"),
    )
    print()
    print("Defined the scheduler.")

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=get_config(section="train", key="batch_size"),
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=get_config(section="train", key="batch_size"),
        shuffle=False,
        collate_fn=collate_fn,
    )
    print()
    print("Defined data loaders.")

    counter = 0
    print("Ready for training")

    for epoch in range(epochs):
        print()
        print(f"Strarted training epoch {epoch}")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            val_loader=val_loader,
        )
        val_loss = validate(
            model=model, dataloader=val_loader, criterion=criterion, device=device
        )

        # Step the scheduler
        scheduler.step(metrics=val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early Stopping and Model Saving logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(
                obj=model.state_dict(),
                f=f"{data_path}/{
                    get_config(section='train', key='model_artifact_file')
                }",
            )
            print(f"  --> New best model saved (Val Loss: {best_val_loss:.6f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
