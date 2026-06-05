import polars as pl


# Mock embed_string
def embed_string(s):
    return [0.1, 0.2, 0.3]


def embed_sequences(sequences_df: pl.DataFrame, all_names: list[str]) -> pl.DataFrame:
    """
    Transforms generated sequences into training data.
    [1] context_names -> multi-hot vector
    [2] context_text -> semantic embedding
    [3] target_name -> one-hot vector
    [4] target_text -> semantic embedding
    """
    main_characters = ["Monica", "Joey", "Chandler", "Phoebe", "Ross", "Rachel"]
    name_list = main_characters + ["Other"]
    name_to_idx = {name: i for i, name in enumerate(name_list)}
    num_names = len(name_list)

    results = []
    for row in sequences_df.iter_rows(named=True):
        # 1. Multi-hot for context names
        context_vec = [0.0] * num_names
        for name in row["context_names"]:
            if name in name_to_idx:
                context_vec[name_to_idx[name]] = 1.0
            else:
                context_vec[name_to_idx["Other"]] = 1.0

        # 2. Embed context text
        context_emb = embed_string(row["context_text"])

        # 3. One-hot for target name
        target_vec = [0.0] * num_names
        target_name = row["target_name"]
        if target_name in name_to_idx:
            target_vec[name_to_idx[target_name]] = 1.0
        else:
            target_vec[name_to_idx["Other"]] = 1.0

        # 4. Embed target text
        target_emb = embed_string(row["target_text"])

        results.append({
            "context_speaker_identity": context_vec,
            "context_text_embedding": context_emb,
            "target_speaker_identity": target_vec,
            "target_text_embedding": target_emb,
        })

    return pl.DataFrame(results)


# Test data
data = {
    "context_names": [["Monica", "Gunther"], ["Joey", "Chandler", "Ross"]],
    "context_text": ["Line 1", "Line 2"],
    "target_name": ["Rachel", "Gunther"],
    "target_text": ["Target 1", "Target 2"],
}
sequences_df = pl.DataFrame(data)

# Run
all_names = ["Monica", "Joey", "Chandler", "Phoebe", "Ross", "Rachel", "Gunther"]
training_df = embed_sequences(sequences_df, all_names)
print(training_df)
