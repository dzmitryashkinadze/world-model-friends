import os
import random

import polars as pl

from world_model_friends.ai.embeddings import embed_string


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


def prepare_training_data(
    sequences_df: pl.DataFrame, all_names: list[str]
) -> pl.DataFrame:
    """
    Transforms generated sequences into training data.
    [1] context_names -> multi-hot vector
    [2] context_text -> semantic embedding
    [3] target_name -> one-hot vector
    [4] target_text -> semantic embedding
    """
    num_names = len(all_names)
    name_to_idx = {name: i for i, name in enumerate(all_names)}

    results = []
    for row in sequences_df.iter_rows(named=True):
        # 1. Multi-hot for context names
        context_vec = [0.0] * num_names
        for name in row["context_names"]:
            if name in name_to_idx:
                context_vec[name_to_idx[name]] = 1.0

        # 2. Embed context text
        context_emb = embed_string(row["context_text"])

        # 3. One-hot for target name
        target_vec = [0.0] * num_names
        if row["target_name"] in name_to_idx:
            target_vec[name_to_idx[row["target_name"]]] = 1.0

        # 4. Embed target text
        target_emb = embed_string(row["target_text"])

        results.append({
            "context_speaker_identity": context_vec,
            "context_text_embedding": context_emb,
            "target_speaker_identity": target_vec,
            "target_text_embedding": target_emb,
        })

    return pl.DataFrame(results)


if __name__ == "__main__":
    # Demonstration
    csv_path = "data/Friends_script.csv"
    if os.path.exists(csv_path):
        # Polars read_csv
        df = pl.read_csv(csv_path)
        print(f"Loaded {csv_path}")
        print(f"Columns: {df.columns}")

        # Generate 10 sequences with max context length 5
        n_val = 10
        k_val = 5
        sequences_df = generate_sequences(df, n_val, k_val)

        print(f"\nGenerated {len(sequences_df)} sequences:")
        print(sequences_df)

        # Demonstrate preparation
        all_names = df["Name"].unique().to_list()
        training_df = prepare_training_data(sequences_df, all_names)
        print("\nPrepared training data (first 2 rows):")
        print(training_df.head(2))
    else:
        print(f"Could not find {csv_path} for demonstration.")
