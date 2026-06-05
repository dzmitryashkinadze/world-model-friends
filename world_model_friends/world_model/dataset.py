import polars as pl
import torch
from torch.utils.data import Dataset


class WorldModelDataset(Dataset):
    """
    PyTorch Dataset for the World Model.
    Expects a Polars DataFrame where columns are sequences (lists/arrays).
    """

    def __init__(self, df: pl.DataFrame):
        self.df = df
        # Store as lists to handle variable lengths
        self.speaker_seqs = df["context_speaker_identity"].to_list()
        self.dialogue_seqs = df["context_text_embedding"].to_list()
        self.target_speakers = df["target_speaker_identity"].to_list()
        self.target_embs = df["target_text_embedding"].to_list()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "speaker_seq": torch.tensor(self.speaker_seqs[idx], dtype=torch.float32),
            "dialogue_seq": torch.tensor(self.dialogue_seqs[idx], dtype=torch.float32),
            "target_speaker": torch.tensor(
                self.target_speakers[idx], dtype=torch.float32
            ),
            "target_emb": torch.tensor(self.target_embs[idx], dtype=torch.float32),
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable sequence lengths.
    """
    speaker_seqs = []
    dialogue_seqs = []
    target_speakers = []
    target_embs = []

    for item in batch:
        speaker_seqs.append(item["speaker_seq"])
        dialogue_seqs.append(item["dialogue_seq"])
        target_speakers.append(item["target_speaker"])
        target_embs.append(item["target_emb"])

    # We need to pad the sequences to the max length in the batch
    # or use a more advanced approach. For simplicity, let's use padding.

    # Pad speaker_seqs
    # speaker_seqs: list of (N, num_speakers)
    max_n = max(s.shape[0] for s in speaker_seqs)
    if speaker_seqs[0].ndim == 1:
        num_speakers = 1
    else:
        num_speakers = speaker_seqs[0].shape[1]

    padded_speaker_seqs = []
    padded_dialogue_seqs = []

    for i in range(len(batch)):
        s = speaker_seqs[i]
        d = dialogue_seqs[i]
        n = s.shape[0]

        # Pad speaker sequence
        s_pad = torch.zeros((max_n, num_speakers))
        if s.ndim == 1:
            s_pad[:n, 0] = s
        else:
            s_pad[:n, :] = s
        padded_speaker_seqs.append(s_pad)

        # Pad dialogue sequence
        if d.ndim == 1:
            emb_dim = 1
            d_pad = torch.zeros((max_n, emb_dim))
            d_pad[:n, 0] = d
        else:
            emb_dim = d.shape[1]
            d_pad = torch.zeros((max_n, emb_dim))
            d_pad[:n, :] = d
        padded_dialogue_seqs.append(d_pad)

    return {
        "speaker_seq": torch.stack(padded_speaker_seqs),
        "dialogue_seq": torch.stack(padded_dialogue_seqs),
        "target_speaker": torch.stack(target_speakers),
        "target_emb": torch.stack(target_embs),
    }
