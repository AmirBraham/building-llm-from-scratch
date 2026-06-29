"""
Chapter 2 data-loading utilities, packaged as an importable module.

    from chapter02 import create_dataloader_v1
"""

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, tokenizer, text, max_length, stride):
        token_ids = torch.tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))

        # X is each window of `max_length` tokens; Y is the same window shifted by one.
        self.input_ids = token_ids[:-1].unfold(0, max_length, stride)
        self.target_ids = token_ids[1:].unfold(0, max_length, stride)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(tokenizer, text, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
