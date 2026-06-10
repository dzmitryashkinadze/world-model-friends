import polars as pl
import torch

from world_model_friends.predictor.dataset import WorldModelDataset, collate_fn


def test_world_model_dataset_init_and_len():
    df = pl.DataFrame({
        "context_identity": [[1.0, 0.0], [0.0, 1.0]],
        "context_embedding": [[0.1, 0.2], [0.3, 0.4]],
        "target_identity": [[0.0, 1.0], [1.0, 0.0]],
        "target_embedding": [[0.5, 0.6], [0.7, 0.8]],
    })
    dataset = WorldModelDataset(df)
    assert len(dataset) == 2


def test_world_model_dataset_getitem():
    df = pl.DataFrame({
        "context_identity": [[1.0, 0.0], [0.0, 1.0]],
        "context_embedding": [[0.1, 0.2], [0.3, 0.4]],
        "target_identity": [[0.0, 1.0], [1.0, 0.0]],
        "target_embedding": [[0.5, 0.6], [0.7, 0.8]],
    })
    dataset = WorldModelDataset(df)
    sample = dataset[0]

    assert isinstance(sample["context_identity"], torch.Tensor)
    assert torch.allclose(
        sample["context_identity"], torch.tensor([1.0, 0.0], dtype=torch.float32)
    )
    assert torch.allclose(
        sample["context_embedding"], torch.tensor([0.1, 0.2], dtype=torch.float32)
    )
    assert torch.allclose(
        sample["target_identity"], torch.tensor([0.0, 1.0], dtype=torch.float32)
    )
    assert torch.allclose(
        sample["target_embedding"], torch.tensor([0.5, 0.6], dtype=torch.float32)
    )


def test_collate_fn():
    batch = [
        {
            "context_identity": torch.tensor([1.0, 0.0]),
            "context_embedding": torch.tensor([0.1, 0.2]),
            "target_identity": torch.tensor([0.0, 1.0]),
            "target_embedding": torch.tensor([0.5, 0.6]),
        },
        {
            "context_identity": torch.tensor([0.0, 1.0]),
            "context_embedding": torch.tensor([0.3, 0.4]),
            "target_identity": torch.tensor([1.0, 0.0]),
            "target_embedding": torch.tensor([0.7, 0.8]),
        },
    ]
    collated = collate_fn(batch)

    assert collated["context_identity"].shape == (2, 2)
    assert collated["context_embedding"].shape == (2, 2)
    assert collated["target_identity"].shape == (2, 2)
    assert collated["target_embedding"].shape == (2, 2)
    assert torch.allclose(collated["context_identity"][0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(collated["context_embedding"][1], torch.tensor([0.3, 0.4]))
