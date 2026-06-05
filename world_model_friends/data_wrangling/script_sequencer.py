import random

import polars as pl
from tqdm import tqdm

from world_model_friends.ai.embeddings import embed_batch, get_model
from world_model_friends.config import get_config


def generate_sequences(
    df: pl.DataFrame, num_sequences: int, max_context_length: int
) -> pl.DataFrame:
    """
    Generates sequences of turns.
    Each sequence has a random length (1 to max_context_length) for context,
    and the next turn as the target.

    df: Polars DataFrame with columns 'Name' and 'Lines'
    num_sequences: Number of sequences to generate
    max_context_length: Max length of context
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


def embed_sequences(sequences_df: pl.DataFrame) -> pl.DataFrame:
    """
    Transforms generated sequences into training data.
    [1] context_text -> semantic embedding
    [2] target_text -> semantic embedding
    """

    # config
    batch_size = get_config("embeddings", "batch_size")

    # get embedding model
    print()
    print("Loading the embedding model:")
    model = get_model()

    # 1. Batch embed all context and target texts
    context_texts = sequences_df["context_text"].to_list()
    target_texts = sequences_df["target_text"].to_list()
    context_embeddings = []
    target_embeddings = []

    print()
    print("Embedding context:")
    for i in tqdm(range(0, len(context_texts), batch_size)):
        context_embeddings += embed_batch(
            model=model, texts=context_texts[i : i + batch_size]
        )

    print()
    print("Embedding targets:")
    for i in tqdm(range(0, len(target_texts), batch_size)):
        target_embeddings += embed_batch(
            model=model, texts=target_texts[i : i + batch_size]
        )

    # 3. Combine everything into a single DataFrame
    return pl.DataFrame({
        "context_identity": sequences_df["context_identity"],
        "context_embedding": context_embeddings,
        "target_identity": sequences_df["target_identity"],
        "target_embedding": target_embeddings,
    })


def split_data(
    df: pl.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Splits the DataFrame into train, test, and validation sets.

    df: Polars DataFrame to split
    train_ratio: Proportion of data for training
    val_ratio: Proportion of data for validation
    (test_ratio will be 1 - train_ratio - val_ratio)
    """
    # Shuffle the data
    df_shuffled = df.sample(fraction=1.0, shuffle=True)

    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_shuffled.slice(0, train_end)
    val_df = df_shuffled.slice(train_end, val_end - train_end)
    test_df = df_shuffled.slice(val_end, n - val_end)
    print()
    print(f"Written {train_df.height} training samples")
    print(f"Written {val_df.height} validation samples")
    print(f"Written {test_df.height} testing samples")

    return train_df, val_df, test_df
