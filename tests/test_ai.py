from world_model_friends.ai import embed_batch, get_model


def test_embed_string_basic():
    """Test the local embedding model."""
    text = "hello world"
    model = get_model()
    embedding = embed_batch(model=model, texts=text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
    assert all(isinstance(x, float) for x in embedding)
