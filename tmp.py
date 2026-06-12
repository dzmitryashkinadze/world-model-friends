from world_model_friends.config import get_config
from world_model_friends.decoder import Decoder

test = Decoder(path=get_config("process", "script_with_line_embeddings_path"))
print(test._embeddings[0])
print(test.search(target=test._embeddings[0], speaker="Joey"))
