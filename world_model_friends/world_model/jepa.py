import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """
    A JEPA-inspired Transformer predictor operating entirely in latent space.

    Instead of flat concatenation, this model projects discrete conditioning
    variables into the latent dimension and processes them as a sequence.
    A learned [PREDICT] token aggregates the context and conditioning
    to output the final target embedding.
    """

    def __init__(
        self,
        num_speakers: int,
        emb_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim

        # 1. Latent Projections
        # Map discrete multi-hot/one-hot vectors into the
        # shared continuous embedding space
        self.context_id_proj = nn.Linear(num_speakers, emb_dim)
        self.target_id_proj = nn.Linear(num_speakers, emb_dim)

        # 2. The Predict Token (analogous to a [CLS] token in BERT)
        # This learned parameter gathers information from the other tokens
        # to form the prediction
        self.predict_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # 3. Transformer Predictor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Final Head
        # Projects the processed predict token into the exact target space
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, emb_dim))

    def forward(
        self,
        context_identities: torch.Tensor,
        context_embedding: torch.Tensor,
        target_identity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            context_identities: (B, num_speakers) - Multi-hot
            context_embedding: (B, emb_dim) - Continuous latent state
            target_identity: (B, num_speakers) - One-hot
        Returns:
            target_embedding: (B, emb_dim) - Predicted next latent state
        """
        batch_size = context_embedding.size(0)

        # Project inputs to tokens of shape (B, 1, emb_dim)
        ctx_id_tok = self.context_id_proj(context_identities).unsqueeze(1)
        ctx_emb_tok = context_embedding.unsqueeze(1)
        tgt_id_tok = self.target_id_proj(target_identity).unsqueeze(1)

        # Expand the predict token for the current batch
        pred_tok = self.predict_token.expand(batch_size, -1, -1)

        # Form the sequence: [Context IDs, Context Emb, Target ID, Predict Token]
        # Shape becomes: (B, 4, emb_dim)
        seq = torch.cat([ctx_id_tok, ctx_emb_tok, tgt_id_tok, pred_tok], dim=1)

        # Process the sequence through the Transformer
        out_seq = self.transformer(seq)

        # Extract the processed Predict Token (which is at the last sequence position)
        pred_out = out_seq[:, -1, :]

        # Output the final embedding
        return self.head(pred_out)
