from unittest.mock import MagicMock, patch

import torch

from world_model_friends.predictor.evaluate import evaluate_world_model
from world_model_friends.predictor.train import train_one_epoch, validate


def test_validate():
    model = MagicMock()

    def side_effect(ctx_id, ctx_emb, tgt_id):
        # Return a tensor that requires grad so it can be used in loss calculation
        return torch.tensor([[0.5, 0.6]], dtype=torch.float32, requires_grad=True)

    model.side_effect = side_effect

    mock_batch = {
        "context_identity": torch.tensor([[1.0, 0.0]]),
        "context_embedding": torch.tensor([[0.1, 0.2]]),
        "target_identity": torch.tensor([[0.0, 1.0]]),
        "target_embedding": torch.tensor([[0.5, 0.6]]),
    }
    dataloader = [mock_batch]
    criterion = torch.nn.MSELoss()
    device = torch.device("cpu")

    avg_loss = validate(model, dataloader, criterion, device)
    assert avg_loss == 0.0


def test_train_one_epoch():
    model = MagicMock()

    def side_effect(ctx_id, ctx_emb, tgt_id):
        return torch.tensor([[0.5, 0.6]], dtype=torch.float32, requires_grad=True)

    model.side_effect = side_effect

    optimizer = MagicMock()
    criterion = torch.nn.MSELoss()

    mock_batch = {
        "context_identity": torch.tensor([[1.0, 0.0]]),
        "context_embedding": torch.tensor([[0.1, 0.2]]),
        "target_identity": torch.tensor([[0.0, 1.0]]),
        "target_embedding": torch.tensor([[0.5, 0.6]]),
    }
    dataloader = [mock_batch]
    val_loader = [mock_batch]
    device = torch.device("cpu")

    with patch("world_model_friends.predictor.train.get_config") as mock_get_config:
        mock_get_config.return_value = 1  # running_train_loss_steps

        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            val_loader=val_loader,
        )

        assert isinstance(avg_loss, float)
        assert optimizer.step.called
        assert optimizer.zero_grad.called


def test_evaluate_side_effects():
    from unittest.mock import MagicMock, patch

    import polars as pl

    with (
        patch("world_model_friends.predictor.evaluate.get_config") as mock_get_config,
        patch("world_model_friends.predictor.evaluate.torch.load") as mock_torch_load,
        patch("world_model_friends.predictor.evaluate.JEPAPredictor") as mock_jepa,
    ):
        # Setup mock return values
        # Need to return something that has len() for num_speakers
        mock_get_config.side_effect = lambda key, subkey, default=None: {
            ("train", "num_heads"): 4,
            ("process", "main_characters"): ["Char1", "Char2"],
            ("embeddings", "dimension"): 16,
            ("train", "num_layers"): 2,
            ("train", "dropout"): 0.1,
            ("train", "batch_size"): 1,
        }.get((key, subkey), default)

        mock_torch_load.return_value = {}
        mock_jepa.return_value.to.return_value = MagicMock()

        # Mock model's forward to return something
        mock_model = mock_jepa.return_value.to.return_value
        mock_model.return_value = torch.tensor(
            [[0.5, 0.6]], dtype=torch.float32, requires_grad=True
        )

        test_df = pl.DataFrame({
            "context_identity": [[1.0, 0.0]],
            "context_embedding": [[0.1, 0.2]],
            "target_identity": [[0.0, 1.0]],
            "target_embedding": [[0.5, 0.6]],
        })

        with patch(
            "world_model_friends.predictor.evaluate.DataLoader"
        ) as mock_dataloader:
            mock_batch = {
                "context_identity": torch.tensor([[1.0, 0.0]]),
                "context_embedding": torch.tensor([[0.1, 0.2]]),
                "target_identity": torch.tensor([[0.0, 1.0]]),
                "target_embedding": torch.tensor([[0.5, 0.6]]),
            }
            mock_dataloader.return_value = [mock_batch]

            evaluate_world_model("dummy_path", test_df, torch.device("cpu"))

    # If it reached here, it means evaluate ran without error.
    pass
