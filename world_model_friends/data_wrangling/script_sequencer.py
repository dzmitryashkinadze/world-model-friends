"""
Module for processing and sequencing data for training.
"""

import random

import polars as pl
from tqdm import tqdm

from world_model_friends.ai.embeddings import embed_batch
from world_model_friends.config import get_config


def process_split(
    split_df: pl.DataFrame, split_name: str, n_sequences: int, max_context_length: int
) -> pl.DataFrame | None:
    """
    Process a dataframe split by generating sequences and embedding them.

    Args:
        split_df (pl.DataFrame): The dataframe split to process.
        split_name (str): Name of the split (e.g., 'train', 'test').
        n_sequences (int): Number of sequences to generate.
        max_context_length (int): Maximum context length for sequences.

    Returns:
        pl.DataFrame | None: The processed DataFrame containing embeddings,
            or None if input is invalid.
    """
    print(f"Processing {split_name} split (target: {n_sequences} sequences)...")
    if len(split_df) == 0 or n_sequences <= 0:
        return None

    # Generate sequences
    seq_df = generate_sequences(split_df, n_sequences, max_context_length)
    print(f"Generated {len(seq_df)} sequences for {split_name}.")

    # Embed sequences
    seq_df = embed_sequences(sequences_df=seq_df, split_name=split_name)
    print(f"Embedded sequences for {split_name}.")
    return seq_df


def generate_sequences(
    df: pl.DataFrame, num_sequences: int, max_context_length: int
) -> pl.DataFrame:
    """
    Generates sequences of turns from a dataframe.
    Each sequence has a random length (1 to max_context_length) for context,
    and the next turn as the target.

    Args:
        df (pl.DataFrame): Polars DataFrame with columns 'Name' and 'Lines'.
        num_sequences (int): Number of sequences to generate.
        max_context_length (int): Max length of context.

    Returns:
        pl.DataFrame: A DataFrame containing the generated sequences and their metadata.
    """
    results = []

    # drop nans
    df = df.drop_nulls(subset=["Lines", "Name"])
    df = df.filter(pl.col("Lines").str.strip_chars() != "")
    df_len = len(df)

    # characters configuration
    name_to_idx = {
        name: i for i, name in enumerate(get_config("process", "main_characters"))
    }
    num_characters = len(get_config("process", "main_characters")) + 1

    # Convert columns to lists for faster access in the loop
    names = df["Name"].to_list()
    lines = df["Lines"].to_list()

    print()
    print("Picking sequences:")
    for _ in tqdm(range(num_sequences)):
        # Random context length cl from 1 to max_context_length
        cl = random.randint(1, max_context_length)

        # The index i must be such that i + cl < df_len
        # So i <= df_len - cl - 1
        max_i = df_len - cl - 1

        if max_i < 0:
            # Adjust cl if it's too large for the current df
            cl = random.randint(1, min(max_context_length, df_len - 1))
            max_i = df_len - cl - 1
            if max_i < 0:
                continue

        i = random.randint(0, max_i)

        # Context rows: df[i : i+cl]
        # target row: df[i+cl]

        context_indices = range(i, i + cl)

        # 1. context_names: list of unique names
        context_identity = [0.0] * num_characters
        for idx in context_indices:
            name = names[idx]
            idx = name_to_idx.get(name, -1)
            if idx != -1:
                context_identity[idx] = 1.0
            else:
                context_identity[-1] = 1.0

        # 2. context_text: "Name: Line\n"
        context_text_parts = []
        for idx in context_indices:
            context_text_parts.append(f"{names[idx]}: {lines[idx]}")
        context_text = "\n".join(context_text_parts)

        # 3. target identity
        target_idx = i + cl
        target_identity = [0.0] * num_characters
        for idx in context_indices:
            name = names[idx]
            idx = name_to_idx.get(name, -1)
            if idx != -1:
                target_identity[idx] = 1.0
            else:
                target_identity[-1] = 1.0

        # 4. target_text
        target_text = lines[target_idx]

        results.append({
            "context_identity": context_identity,
            "context_text": context_text,
            "target_identity": target_identity,
            "target_text": target_text,
            "context_length": cl,
        })

    return pl.DataFrame(results)


def embed_sequences(sequences_df: pl.DataFrame, split_name: str) -> None:
    """
    Transforms generated sequences into training data by creating semantic embeddings.

    Args:
        sequences_df (pl.DataFrame): The generated sequences DataFrame.
        split_name (str): Name of the split for output filenames.

    Returns:
        None: This function writes parquet files to the 'data/' directory.
    """

    # config
    batch_size = get_config("embeddings", "batch_size")

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
    for i in tqdm(range(0, len(context_texts), batch_size)):
        context_embeddings = embed_batch(texts=context_texts[i : i + batch_size])
        target_embeddings = embed_batch(texts=target_texts[i : i + batch_size])

        # store the chunk
        pl.DataFrame({
            "context_identity": context_identities[i : i + batch_size],
            "context_embedding": context_embeddings,
            "target_identity": target_identities[i : i + batch_size],
            "target_embedding": target_embeddings,
        }).write_parquet(f"data/{split_name}_{i}.parquet")


def split_raw_data(
    df: pl.DataFrame, test_ratio: float = 0.1, val_ratio: float = 0.1
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Splits the DataFrame into test, validation, and training sets sequentially.

    Args:
        df (pl.DataFrame): The Polars DataFrame to split.
        test_ratio (float): Proportion of data for testing. Defaults to 0.1.
        val_ratio (float): Proportion of data for validation. Defaults to 0.1.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
            A tuple of (test_df, val_df, train_df).
    """
    n = len(df)
    test_end = int(n * test_ratio)
    val_end = int(n * (test_ratio + val_ratio))

    test_df = df.slice(0, test_end)
    val_df = df.slice(test_end, val_end - test_end)
    train_df = df.slice(val_end, n - val_end)

    print()
    print("Split raw data sequentially:")
    print(f"Test={test_df.height}")
    print(f"Val={val_df.height}")
    print(f"Train={train_df.height}")

    return test_df, val_df, train_df
