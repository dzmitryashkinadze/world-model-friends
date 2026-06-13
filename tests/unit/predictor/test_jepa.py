import torch

from world_model_friends.predictor.jepa import JEPAPredictor


def test_jepa_predictor_init():
    num_speakers = 5
    emb_dim = 16
    model = JEPAPredictor(num_speakers=num_speakers, emb_dim=emb_dim)
    assert model.emb_dim == emb_dim
    assert isinstance(model.context_id_proj, torch.nn.Linear)
    assert isinstance(model.target_id_proj, torch.nn.Linear)
    assert isinstance(model.predict_token, torch.nn.Parameter)


def test_jepa_predictor_forward():
    num_speakers = 5
    emb_dim = 16
    batch_size = 4
    model = JEPAPredictor(num_speakers=num_speakers, emb_dim=emb_dim)
    model.eval()

    context_identity = torch.zeros(batch_size, num_speakers)  # Simplified for test
    context_embedding = torch.randn(batch_size, emb_dim)
    target_identity = torch.zeros(batch_size, num_speakers)  # Simplified for test

    with torch.no_grad():
        output = model(context_identity, context_embedding, target_identity)

    assert output.shape == (batch_size, emb_dim)
