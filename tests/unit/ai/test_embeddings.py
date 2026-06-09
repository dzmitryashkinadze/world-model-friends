from unittest.mock import MagicMock, patch

import pytest

from world_model_friends.ai import embed_batch


def test_embed_batch_success():
    """Test embed_batch with a mocked response."""
    mock_response_data = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
    }

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        texts = ["text1", "text2"]
        embeddings = embed_batch(texts=texts)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

        # Verify the call
        args, kwargs = mock_post.call_args
        assert kwargs["json"] == {"input": texts}


def test_embed_batch_failure():
    """Test embed_batch when requests.post fails."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Connection error")
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            embed_batch(texts=["test"])
        assert "Connection error" in str(excinfo.value)
