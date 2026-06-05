from sentence_transformers import SentenceTransformer

from world_model_friends.config import get_config


def get_model():
    """Get global model."""
    return SentenceTransformer(get_config("embeddings", "model_name"))


def embed_batch(model, texts: list[str]) -> list[list[float]]:
    """Embeds a list of strings using the configured model."""
    embeddings = model.encode(inputs=texts)
    return embeddings.tolist()
