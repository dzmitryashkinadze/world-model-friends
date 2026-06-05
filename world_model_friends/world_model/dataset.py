import polars as pl
import torch
from torch.utils.data import Dataset


class WorldModelDataset(Dataset):
    """
    PyTorch Dataset for the World Model.
    Expects a Polars DataFrame where columns are sequences (lists/arrays).
    """

    def __init__(self, df: pl.DataFrame):
        self.df = df
        # Store as lists to handle variable lengths
        self.context_identity = df["context_identity"].to_list()
        self.context_embedding = df["context_embedding"].to_list()
        self.target_identity = df["target_identity"].to_list()
        self.target_embedding = df["target_embedding"].to_list()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "context_identity": torch.tensor(
                self.context_identity[idx], dtype=torch.float32
            ),
            "context_embedding": torch.tensor(
                self.context_embedding[idx], dtype=torch.float32
            ),
            "target_identity": torch.tensor(
                self.target_identity[idx], dtype=torch.float32
            ),
            "target_embedding": torch.tensor(
                self.target_embedding[idx], dtype=torch.float32
            ),
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable sequence lengths.
    """
    context_identity = []
    context_embedding = []
    target_identity = []
    target_embedding = []

    for item in batch:
        context_identity.append(item["context_identity"])
        context_embedding.append(item["context_embedding"])
        target_identity.append(item["target_identity"])
        target_embedding.append(item["target_embedding"])
    return {
        "context_identity": torch.stack(context_identity),
        "context_embedding": torch.stack(context_embedding),
        "target_identity": torch.stack(target_identity),
        "target_embedding": torch.stack(target_embedding),
    }
