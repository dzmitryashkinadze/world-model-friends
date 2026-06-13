"""Single-object inference for the JEPA Predictor."""

import torch

from world_model_friends.predictor.jepa import JEPAPredictor


def infer(
    request: tuple[list[float], list[float], list[float]],
    model: JEPAPredictor,
) -> torch.Tensor:
    """Run a single forward pass of the JEPA model on an inference request.

    Args:
        request: Tuple of (context_identity, context_embedding, target_identity)
            as returned by ``embed_inference_request()``.
        model: A trained :class:`JEPAPredictor` instance.

    Returns:
        torch.Tensor: Predicted target embedding of shape ``(1, emb_dim)``.
    """
    context_identity, context_embedding, target_identity = request

    # Convert lists to tensors and add batch dimension
    context_identity = torch.tensor(context_identity, dtype=torch.float32).unsqueeze(0)
    context_embedding = torch.tensor(context_embedding, dtype=torch.float32).unsqueeze(
        0
    )
    target_identity = torch.tensor(target_identity, dtype=torch.float32).unsqueeze(0)

    # Forward pass
    prediction = model(context_identity, context_embedding, target_identity)

    return prediction
