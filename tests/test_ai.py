from world_model_friends.ai import embed_string


def test_embed_string_basic():
    """Test the local embedding model."""
    text = "hello world"
    embedding = embed_string(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
    assert all(isinstance(x, float) for x in embedding)
