import random

import polars as pl

from world_model_friends.ai.embeddings import embed_string
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
    df_len = len(df)

    # Convert columns to lists for faster access in the loop
    names = df["Name"].to_list()
    lines = df["Lines"].to_list()

    for _ in range(num_sequences):
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
        context_names_list = []
        seen_names = set()
        for idx in context_indices:
            name = names[idx]
            if name not in seen_names:
                context_names_list.append(name)
                seen_names.add(name)

        # 2. context_text: "Name: Line\n"
        context_text_parts = []
        for idx in context_indices:
            context_text_parts.append(f"{names[idx]}: {lines[idx]}")
        context_text = "\n".join(context_text_parts)

        # 3. target_name
        target_idx = i + cl
        target_name = names[target_idx]

        # 4. target_text
        target_text = lines[target_idx]

        results.append({
            "context_names": context_names_list,
            "context_text": context_text,
            "target_name": target_name,
            "target_text": target_text,
            "context_length": cl,
        })

    return pl.DataFrame(results)


def embed_sequences(sequences_df: pl.DataFrame) -> pl.DataFrame:
    """
    Transforms generated sequences into training data.
    [1] context_names -> multi-hot vector
    [2] context_text -> semantic embedding
    [3] target_name -> one-hot vector
    [4] target_text -> semantic embedding
    """
    name_to_idx = {
        name: i for i, name in enumerate(get_config("process", "main_characters"))
    }
    num_characters = len(get_config("process", "main_characters")) + 1

    results = []
    for row in sequences_df.iter_rows(named=True):
        # 1. Multi-hot for context names
        context_vec = [0.0] * num_characters
        for name in row["context_names"]:
            if name in name_to_idx:
                context_vec[name_to_idx[name]] = 1.0
            else:
                context_vec[-1] = 1.0

        # 2. Embed context text
        context_emb = embed_string(row["context_text"])

        # 3. One-hot for target name
        target_vec = [0.0] * num_characters
        target_name = row["target_name"]
        if target_name in name_to_idx:
            target_vec[name_to_idx[target_name]] = 1.0
        else:
            target_vec[-1] = 1.0

        # 4. Embed target text
        target_emb = embed_string(row["target_text"])

        results.append({
            "context_identity": context_vec,
            "context_embedding": context_emb,
            "target_identity": target_vec,
            "target_embedding": target_emb,
        })

    return pl.DataFrame(results)


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

    return train_df, val_df, test_df
