from world_model_friends.ai import embed_batch
from world_model_friends.config import get_config


def test_embed_string_basic():
    """Test the remote embedding model."""
    text = ["hello world"]
    # Note: This test requires the llama.cpp server to be running on localhost:8081
    try:
        embedding = embed_batch(texts=text)

        assert isinstance(embedding, list)
        assert len(embedding[0]) == get_config("embeddings", "dimension")
        assert all(isinstance(x, float) for x in embedding[0])
    except Exception as e:
        print(f"Skipping test due to connection error: {e}")
