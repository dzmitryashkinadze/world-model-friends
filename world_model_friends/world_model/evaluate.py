import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from world_model_friends.config import get_config
from world_model_friends.world_model.dataset import WorldModelDataset, collate_fn
from world_model_friends.world_model.jepa import JEPAPredictor


def evaluate_world_model(
    model_path: str,
    test_df: pl.DataFrame,
    device: torch.device,
) -> None:
    """
    Evaluates a trained model on a given dataset.

    Args:
        model_path (str): Path to the PyTorch model artifact (.pt).
        test_df (pl.DataFrame): The test dataset as a Polars DataFrame.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: The average loss for the test set.
    """
    # 1. Get config for model architecture
    num_heads = get_config("train", "num_heads")
    num_speakers = len(get_config("process", "main_characters")) + 1
    emb_dim = get_config("embeddings", "dimension")
    num_layers = get_config("train", "num_layers", default=2)
    dropout = get_config("train", "dropout")

    # 2. Instantiate model
    model = JEPAPredictor(
        num_speakers=num_speakers,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # 3. Load weights
    print(f"Loading model weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Prepare dataset
    val_ds = WorldModelDataset(test_df)
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=get_config("train", "batch_size"),
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 5. Evaluate
    criterion_huber = nn.HuberLoss()
    criterion_mse = nn.MSELoss()
    total_loss_huber = 0
    total_loss_mse = 0
    print("Starting evaluation...")
    with torch.no_grad():
        print()
        print("TQDM test batches:")
        for batch in tqdm(val_loader):
            context_identity = batch["context_identity"].to(device)
            context_embedding = batch["context_embedding"].to(device)
            target_identity = batch["target_identity"].to(device)
            target_embedding = batch["target_embedding"].to(device)

            preds = model(context_identity, context_embedding, target_identity)
            loss_huber = criterion_huber(preds, target_embedding)
            loss_mse = criterion_mse(preds, target_embedding)
            total_loss_huber += loss_huber.item()
            total_loss_mse += loss_mse.item()

    avg_loss_huber = total_loss_huber / len(val_loader)
    avg_loss_mse = total_loss_mse / len(val_loader)
    print(f"Evaluation completed. Average Huber loss: {avg_loss_huber:.6f}")
    print(f"Evaluation completed. Average MSE loss: {avg_loss_mse:.6f}")
