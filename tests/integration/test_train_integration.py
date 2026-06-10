import os
from unittest.mock import patch

import polars as pl

from world_model_friends.predictor.train import train_world_model


def test_train_world_model_integration(tmp_path):
    # [0] Setup test data
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # We'll use dim = 8
    dim = 8

    # Create dummy dataframes
    # Columns: context_identity, context_embedding, target_identity, target_embedding
    # Identities must be one-hot/multi-hot vectors of shape (num_speakers,)

    train_df = pl.DataFrame({
        "context_identity": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
        "context_embedding": [[0.1] * dim for _ in range(3)],
        "target_identity": [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]],
        "target_embedding": [[0.2] * dim for _ in range(3)],
    })
    val_df = pl.DataFrame({
        "context_identity": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
        "context_embedding": [[0.1] * dim for _ in range(2)],
        "target_identity": [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]],
        "target_embedding": [[0.2] * dim for _ in range(2)],
    })

    # Mock config dictionary.
    # num_speakers = len(main_characters) + 1
    # We'll set 4 characters, so num_speakers = 5.

    mock_config = {
        "train": {
            "epochs": 1,
            "num_heads": 2,  # Must be <= emb_dim and divides it
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "scheduler_mode": "min",
            "scheduler_factor": 0.5,
            "scheduler_patience": 1,
            "patience": 1,
            "batch_size": 2,
            "max_files": 1,
            "running_train_loss_steps": 1,
        },
        "embedding": {
            "dimension": dim,
        },
        "process": {
            "main_characters": ["Alice", "Bob", "Charlie", "David"],
        },
    }

    def side_effect(section, key, default=None):
        try:
            return mock_config[section][key]
        except KeyError:
            return default

    with patch(
        "world_model_friends.predictor.train.get_config", side_effect=side_effect
    ):
        # [2] Run the 'train' command
        train_world_model(train_df=train_df, val_df=val_df)

    # [4] Verify model file was created
    assert os.path.exists("data/best_model.pt")
    # Clean up
    os.remove("data/best_model.pt")
