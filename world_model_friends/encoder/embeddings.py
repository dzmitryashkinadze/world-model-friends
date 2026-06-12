"""
This module provides functionality for generating embeddings
for text using a remote server.
"""

import polars as pl
from tqdm import tqdm

from world_model_friends.config import get_config
from world_model_friends.encoder.model import model


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of strings using a remote llama.cpp server.

    Args:
        texts (list[str]): A list of strings to be embedded.

    Returns:
        list[list[float]]: A list of embeddings,
            where each embedding is a list of floats.
    """
    embeddings = model.encode(texts=texts, convert_to_numpy=True)

    # Convert embeddings to list of lists
    return embeddings.tolist()


def embed_lines(df: pl.DataFrame) -> pl.DataFrame:
    """
    Embeds the 'Lines' column of a CSV file and saves it to a Parquet file.
    """

    # Get embeddings for 'Lines'
    embeddings = model.encode(texts=df["Lines"].to_list(), convert_to_numpy=True)

    # Convert embeddings to list of lists for Polars
    embeddings_list = embeddings.tolist()

    # Add embeddings column
    return df.with_columns(pl.Series("embeddings", embeddings_list))


def embed_sequences(
    sequences_df: pl.DataFrame, split_name: str, output_dir: str = "data"
) -> pl.DataFrame:
    """
    Transforms generated sequences into training data by creating semantic embeddings.

    Args:
        sequences_df (pl.DataFrame): The generated sequences DataFrame.
        split_name (str): Name of the split for output filenames.
        output_dir (str): Directory to save the processed parquet files
            Defaults to 'data'.

    Returns:
        pl.DataFrame: The processed DataFrame containing embeddings.
    """

    # config
    batch_size = get_config("embedding", "batch_size")

    # get embedding model
    print()
    print("Loading the embedding model:")

    # 1. Batch embed all context and target texts
    context_texts = sequences_df["context_text"].to_list()
    target_texts = sequences_df["target_text"].to_list()
    context_identities = sequences_df["context_identity"].to_list()
    target_identities = sequences_df["target_identity"].to_list()

    print()
    print("Embedding chunks:")
    all_chunks = []
    for i in tqdm(range(0, len(context_texts), batch_size)):
        context_embeddings = embed_batch(texts=context_texts[i : i + batch_size])
        target_embeddings = embed_batch(texts=target_texts[i : i + batch_size])

        # store the chunk
        chunk = pl.DataFrame({
            "context_identity": context_identities[i : i + batch_size],
            "context_embedding": context_embeddings,
            "target_identity": target_identities[i : i + batch_size],
            "target_embedding": target_embeddings,
        })
        chunk.write_parquet(f"{output_dir}/{split_name}_{i}.parquet")
        all_chunks.append(chunk)

    return pl.concat(all_chunks)
