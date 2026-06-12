"""Decoder module for converting predicted embeddings into human-readable dialogue."""

from __future__ import annotations

import numpy as np
import polars as pl


class Decoder:
    """Decode a predicted embedding into dialogue lines."""

    def __init__(self, path: str, top_k: int = 3) -> None:
        self._path = path
        self._top_k = top_k
        self.lines_df: pl.DataFrame = pl.read_parquet(path)
        self._embeddings = np.stack(self.lines_df["line_embedding"].to_list())

    def search(self, target: np.ndarray, speaker: str | None = None) -> list[dict]:
        """Return the top-k most similar lines to the target embedding."""
        # Compute cosine similarity for all embeddings
        norms = np.linalg.norm(self._embeddings, axis=-1)
        target_norm = np.linalg.norm(target)
        similarities = self._embeddings.dot(target) / (norms * target_norm + 1e-8)

        if speaker:
            mask = np.array([r == speaker for r in self.lines_df["Name"].to_list()])
            similarities = np.where(mask, similarities, -1.0)

        # Get top-k indices
        indices = np.argsort(-similarities)[: self._top_k]

        results = []
        for idx in indices:
            if similarities[idx] < 0:
                continue
            row = self.lines_df.slice(idx, 1)
            results.append({
                "Name": row["Name"].to_list()[0],
                "Lines": row["Lines"].to_list()[0],
                "cosine_similarity": float(similarities[idx]),
            })

        return results
