from sentence_transformers import SentenceTransformer

from world_model_friends.config import get_config


# wrapper for lazy loading of the model
# so that integration tests run faster
class EmbeddingModel:
    def __init__(self) -> None:
        # lazy loading
        self.model = None

    def encode(self, texts, *args, **kwargs):
        """Encode texts."""
        # load the model only if actually invoked
        if self.model is None:
            print("Loading the embedding model from HF")
            self.model = SentenceTransformer(
                model_name_or_path=get_config(section="embedding", key="model_name")
            )

        return self.model.encode(texts, *args, **kwargs)


model = EmbeddingModel()
