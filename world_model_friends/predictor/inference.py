"""Single-object inference for the JEPA Predictor."""

import numpy as np
import torch

from world_model_friends.config import get_config
from world_model_friends.predictor.jepa import JEPAPredictor


def infer(
    request: tuple[list[float], list[float], list[float]],
    model_path: str,
    data_path: str,
) -> np.ndarray:
    """Run a single forward pass of the JEPA model on an inference request.

    Args:
        request: Tuple of (context_identity, context_embedding, target_identity)
            as returned by ``embed_inference_request()``.
        model: A trained :class:`JEPAPredictor` instance.
        data_path: Path to the data folder.

    Returns:
        np.ndarray: Predicted target embedding of shape ``(1, emb_dim)``.
    """

    # 1. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get config for model architecture
    n_heads = get_config(section="train", key="n_heads")
    n_speakers = len(get_config(section="process", key="main_characters")) + 1
    emb_dim = get_config(section="embedding", key="dimension")
    n_layers = get_config(section="train", key="n_layers", default=2)
    dropout = get_config(section="train", key="dropout")

    model = JEPAPredictor(
        n_speakers=n_speakers,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    # Load weights
    state_dict = torch.load(f=f"{data_path}/{model_path}", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    context_identity, context_embedding, target_identity = request

    # Convert lists to tensors and add batch dimension
    context_identity = torch.tensor(
        data=context_identity, dtype=torch.float32
    ).unsqueeze(dim=0)
    context_embedding = torch.tensor(
        data=context_embedding, dtype=torch.float32
    ).unsqueeze(dim=0)
    target_identity = torch.tensor(data=target_identity, dtype=torch.float32).unsqueeze(
        dim=0
    )

    # Forward pass
    prediction = model(context_identity, context_embedding, target_identity)

    return prediction.detach().cpu().numpy().squeeze()
