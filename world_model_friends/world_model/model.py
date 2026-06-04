import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """
    A Transformer-based world model that predicts the next semantic embedding.

    Inputs:
        speaker_seq: (batch, seq_len, num_speakers) - One-hot speaker identities
        dialogue_seq: (batch, seq_len, emb_dim) - Semantic embeddings
        target_speaker: (batch, num_speakers) - One-hot target speaker identity

    Output:
        predicted_emb: (batch, emb_dim) - Predicted next semantic embedding
    """

    def __init__(
        self,
        num_speakers: int,
        emb_dim: int,
        seq_len: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_speakers = num_speakers
        self.emb_dim = emb_dim
        self.seq_len = seq_len

        # Input dimension is num_speakers + emb_dim
        self.input_dim = num_speakers + emb_dim

        # Project the concatenated input to emb_dim
        self.input_projection = nn.Linear(self.input_dim, emb_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        # We combine the transformer output (last token) with the target_speaker
        self.output_projection = nn.Sequential(
            nn.Linear(emb_dim + num_speakers, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(
        self,
        speaker_seq: torch.Tensor,
        dialogue_seq: torch.Tensor,
        target_speaker: torch.Tensor,
    ) -> torch.Tensor:
        # speaker_seq: (B, N, num_speakers)
        # dialogue_seq: (B, N, emb_dim)
        # target_speaker: (B, num_speakers)

        # 1. Concatenate speaker and dialogue sequences
        # (B, N, num_speakers + emb_dim)
        combined_seq = torch.cat([speaker_seq, dialogue_seq], dim=-1)

        # 2. Project to emb_dim
        # (B, N, emb_dim)
        x = self.input_projection(combined_seq)

        # 3. Pass through transformer
        # (B, N, emb_dim)
        x = self.transformer(x)

        # 4. Use the last token's representation
        # (B, emb_dim)
        last_token = x[:, -1, :]

        # 5. Combine with target_speaker
        # (B, emb_dim + num_speakers)
        combined_out = torch.cat([last_token, target_speaker], dim=-1)

        # 6. Predict target embedding
        # (B, emb_dim)
        return self.output_projection(combined_out)
