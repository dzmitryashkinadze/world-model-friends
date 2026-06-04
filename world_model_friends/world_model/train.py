import numpy as np
import polars as pl
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

        speaker_seq = batch["speaker_seq"].to(device)
        dialogue_seq = batch["dialogue_seq"].to(device)
        target_speaker = batch["target_speaker"].to(device)
        target_emb = batch["target_emb"].to(device)

        # Forward pass
        # Note: The model implementation might need to handle the padding
        # if we are using it, but for a simple draft, let's assume it works.
        preds = model(speaker_seq[:, -1, :], dialogue_seq[:, -1, :], target_speaker)

        loss = criterion(preds, target_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            speaker_seq = batch["speaker_seq"].to(device)
            dialogue_seq = batch["dialogue_seq"].to(device)
            target_speaker = batch["target_speaker"].to(device)
            target_emb = batch["target_emb"].to(device)
            preds = model(speaker_seq[:, -1, :], dialogue_seq[:, -1, :], target_speaker)
            loss = criterion(preds, target_emb)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main(training_df, validation_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine dimensions from the data
    # We assume the data is in the correct sequence format now.
    # For the sake of this draft, we'll extract them from the first element.

    train_ds = WorldModelDataset(training_df)
    val_ds = WorldModelDataset(validation_df)

    # We need to know num_speakers and emb_dim.
    # Let's assume they are in the data.
    # Since we don't have the actual data, let's use some defaults
    # or try to infer.

    # To infer:
    # num_speakers is the length of the target_speaker vector
    # emb_dim is the length of the target_emb vector

    # Let's use a sample from the dataset to get dimensions
    sample_item = train_ds[0]
    num_speakers = sample_item["target_speaker"].shape[0]
    emb_dim = sample_item["target_emb"].shape[0]
    # The sequence length N is not strictly needed for the model if it's dynamic,
    # but for the transformer we need to know it's the max in the batch.
    # Let's just use a large enough value or let the model handle it.
    # In the model, seq_len is used for initialization but not strictly for forward.
    # Actually, it's not used in my model's __init__ except as a param.

    model = WorldModel(
        num_speakers=num_speakers,
        emb_dim=emb_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    epochs = 10
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # In a real script, we would save the model here.
            # torch.save(model.state_dict(), "best_model.pt")


if __name__ == "__main__":
    # For testing purposes, we'll use dummy data.
    # In production, this will be called by the CLI.

    # Create dummy Polars DataFrames
    num_samples = 100
    num_speakers = 10
    emb_dim = 384
    max_seq = 5

    def create_dummy_df(n, num_s, dim, max_seq):
        rows = []
        for _ in range(n):
            seq_len = np.random.randint(1, max_seq + 1)
            rows.append({
                "context_speaker_identity": np.random.randn(seq_len, num_s).tolist(),
                "context_text_embedding": np.random.randn(seq_len, dim).tolist(),
                "target_speaker_identity": np.random.randn(num_s).tolist(),
                "target_text_embedding": np.random.randn(dim).tolist(),
            })
        return pl.DataFrame(rows)

    df_train = create_dummy_df(80, num_speakers, emb_dim, max_seq)
    df_val = create_dummy_df(20, num_speakers, emb_dim, max_seq)

    main(df_train, df_val)
