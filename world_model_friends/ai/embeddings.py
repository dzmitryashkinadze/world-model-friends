from sentence_transformers import SentenceTransformer

# Global model instance to avoid reloading on every call
_model = None


def _get_model():
    """Get global model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_string(text: str) -> list[float]:
    """Embeds a string using all-MiniLM-L6-v2."""
    model = _get_model()
    embedding = model.encode(text)
    return embedding.tolist()
