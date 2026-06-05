import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """
    A model that predicts the next semantic embedding.

    Inputs:
        context_identities: (batch, num_speakers) - Multi-hot vector
            of speakers in context
        context_embedding: (batch, emb_dim) - The current semantic embedding
        target_identity: (batch, num_speakers) - One-hot target speaker identity

    Output:
        target_embedding: (batch, emb_dim) - Predicted next semantic embedding
    """

    def __init__(
        self,
        num_speakers: int,
        emb_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Input dimension: context_identities (num_speakers) +
        # context_embedding (emb_dim) + target_identity (num_speakers)
        self.input_dim = num_speakers + emb_dim + num_speakers

        # MLP architecture to process the latent representation
        layers = []
        curr_dim = self.input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, emb_dim * 2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = emb_dim * 2

        layers.append(nn.Linear(curr_dim, emb_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        context_identities: torch.Tensor,
        context_embedding: torch.Tensor,
        target_identity: torch.Tensor,
    ) -> torch.Tensor:
        # context_identities: (B, num_speakers) - multi-hot
        # context_embedding: (B, emb_dim)
        # target_identity: (B, num_speakers) - one-hot

        # Concatenate all three inputs along the feature dimension
        x = torch.cat([context_identities, context_embedding, target_identity], dim=-1)

        # Predict target embedding
        return self.mlp(x)
