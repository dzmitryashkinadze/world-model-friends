from sentence_transformers import SentenceTransformer

from world_model_friends.config import get_config

# Global model instance to avoid reloading on every call
_model = None


def _get_model():
    """Get global model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(get_config("embeddings", "model_name"))
    return _model


def embed_string(text: str) -> list[float]:
    """Embeds a string using the configured model."""
    model = _get_model()
    embedding = model.encode(text)
    return embedding.tolist()
