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
    return df.with_columns(pl.Series("line_embedding", embeddings_list))


def embed_sequences(
    sequences_df: pl.DataFrame, split_name: str, output_dir: str = "data"
) -> None:
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
    context_identities = sequences_df["context_identity"].to_list()
    target_identities = sequences_df["target_identity"].to_list()
    target_embeddings = sequences_df["target_embedding"].to_list()

    print()
    print("Embedding chunks:")
    for i in tqdm(range(0, len(context_texts), batch_size)):
        context_embeddings = embed_batch(texts=context_texts[i : i + batch_size])

        # store the chunk
        chunk = pl.DataFrame({
            "context_identity": context_identities[i : i + batch_size],
            "context_embedding": context_embeddings,
            "target_identity": target_identities[i : i + batch_size],
            "target_embedding": target_embeddings[i : i + batch_size],
        })
        chunk.write_parquet(f"{output_dir}/{split_name}_{i}.parquet")


def embed_inference_request(
    context_identities: list[str], context_text: str, target_identity: str
) -> tuple[list[float], list[float], list[float]]:
    """
    Embed an inference request by converting identity names to vectors
    and embedding the context text.

    Args:
        context_identities (list[str]): List of character names in the context.
        context_text (str): The context text to embed.
        target_identity (str): The target character name.

    Returns:
        tuple[list[float], list[float], list[float]]:
            (context_identity, context_embedding, target_identity) as vectors.
    """

    # get main characters configuration
    main_characters = get_config("process", "main_characters")
    num_speakers = len(main_characters) + 1  # +1 for unknown characters

    # build name-to-index mapping
    name_to_idx = {name: i for i, name in enumerate(main_characters)}

    # 1. Convert context identities to multi-hot vector
    context_identity = [0.0] * num_speakers
    for name in context_identities:
        idx = name_to_idx.get(name, -1)
        if idx != -1:
            context_identity[idx] = 1.0
        else:
            context_identity[-1] = 1.0

    # 2. Embed the context text
    context_embedding = embed_batch([context_text])[0]

    # 3. Convert target identity to one-hot vector
    target_vec = [0.0] * num_speakers
    idx = name_to_idx.get(target_identity, -1)
    if idx != -1:
        target_vec[idx] = 1.0
    else:
        target_vec[-1] = 1.0

    return (context_identity, context_embedding, target_vec)
