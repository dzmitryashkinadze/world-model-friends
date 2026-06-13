"""Single-object inference for the JEPA Predictor."""

import numpy as np
import torch

from world_model_friends.config import get_config
from world_model_friends.predictor.jepa import JEPAPredictor


def infer(
    request: tuple[list[float], list[float], list[float]], model_path: str
) -> np.ndarray:
    """Run a single forward pass of the JEPA model on an inference request.

    Args:
        request: Tuple of (context_identity, context_embedding, target_identity)
            as returned by ``embed_inference_request()``.
        model: A trained :class:`JEPAPredictor` instance.

    Returns:
        np.ndarray: Predicted target embedding of shape ``(1, emb_dim)``.
    """

    # 1. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get config for model architecture
    num_heads = get_config("train", "num_heads")
    num_speakers = len(get_config("process", "main_characters")) + 1
    emb_dim = get_config("embedding", "dimension")
    num_layers = get_config("train", "num_layers", default=2)
    dropout = get_config("train", "dropout")

    model = JEPAPredictor(
        num_speakers=num_speakers,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    context_identity, context_embedding, target_identity = request

    # Convert lists to tensors and add batch dimension
    context_identity = torch.tensor(context_identity, dtype=torch.float32).unsqueeze(0)
    context_embedding = torch.tensor(context_embedding, dtype=torch.float32).unsqueeze(
        0
    )
    target_identity = torch.tensor(target_identity, dtype=torch.float32).unsqueeze(0)

    # Forward pass
    prediction = model(context_identity, context_embedding, target_identity)

    return prediction.detach().cpu().numpy().squeeze()
