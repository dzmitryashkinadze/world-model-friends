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
    return df.with_columns(pl.Series(name="line_embedding", values=embeddings_list))


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
    batch_size = get_config(section="embedding", key="batch_size")

    # get embedding model
    print()
    print("Loading the embedding model:")

    # 1. Batch embed all context and target texts
    context_texts = sequences_df["context_text"].to_list()
    context_identity = sequences_df["context_identity"].to_list()
    target_identity = sequences_df["target_identity"].to_list()
    target_embeddings = sequences_df["target_embedding"].to_list()

    print()
    print("Embedding chunks:")
    for i in tqdm(range(0, len(context_texts), batch_size)):
        context_embeddings = embed_batch(texts=context_texts[i : i + batch_size])

        # store the chunk
        chunk = pl.DataFrame(
            data={
                "context_identity": context_identity[i : i + batch_size],
                "context_embedding": context_embeddings,
                "target_identity": target_identity[i : i + batch_size],
                "target_embedding": target_embeddings[i : i + batch_size],
            }
        )
        chunk.write_parquet(file=f"{output_dir}/{split_name}_{i}.parquet")


def embed_names(names: list[str]) -> list[float]:
    """Embed character names (Joey, Monica, etc) into a multi-hot vector."""
    # get main characters configuration
    main_characters = get_config(section="process", key="main_characters")
    n_speakers = len(main_characters) + 1  # +1 for unknown characters

    # build name-to-index mapping
    name_to_idx = {name: i for i, name in enumerate(main_characters)}

    # 1. Convert context identities to multi-hot vector
    context_identity = [0.0] * n_speakers
    for name in names:
        idx = name_to_idx.get(name, -1)
        if idx != -1:
            context_identity[idx] = 1.0
        else:
            context_identity[-1] = 1.0
    return context_identity


def embed_inference_request(
    context_names: list[str], context_text: str, target_name: str
) -> tuple[list[float], list[float], list[float]]:
    """
    Embed an inference request by converting identity names to vectors
    and embedding the context text.

    Args:
        context_names (list[str]): List of character names in the context.
        context_text (str): The context text to embed.
        target_name (str): The target character name.

    Returns:
        tuple[list[float], list[float], list[float]]:
            (context_identity, context_embedding, target_identity) as vectors.
    """
    # 1. Embed identities
    context_identity = embed_names(names=context_names)
    target_identity = embed_names(names=[target_name])

    # 2. Embed the context text
    context_embedding = embed_batch(texts=[context_text])[0]

    return (context_identity, context_embedding, target_identity)
