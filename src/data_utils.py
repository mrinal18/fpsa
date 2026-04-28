import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GlueDataset(Dataset):
    def __init__(self, split, field1, field2):
        self.split = split
        self.field1 = field1
        self.field2 = field2

    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        ex = self.split[i]
        if self.field2:
            enc = tokenizer(ex[self.field1], ex[self.field2],
                            truncation=True, padding="max_length",
                            max_length=MAX_LEN, return_tensors="pt")
        else:
            enc = tokenizer(ex[self.field1],
                            truncation=True, padding="max_length",
                            max_length=MAX_LEN, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.long),
        }

class MLMDataset(Dataset):
    """Chunks a flat list of token ids into fixed-length blocks with MLM masking."""
    def __init__(self, token_ids, block_size=128, mask_prob=0.15, seed=0):
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.cls = tokenizer.cls_token_id
        self.sep = tokenizer.sep_token_id
        self.mask = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        # Leave room for [CLS] and [SEP]
        chunk_content = block_size - 2
        n_chunks = len(token_ids) // chunk_content
        self.chunks = []
        for i in range(n_chunks):
            chunk = token_ids[i*chunk_content:(i+1)*chunk_content]
            self.chunks.append(chunk)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        chunk = self.chunks[i]
        ids = [self.cls] + chunk + [self.sep]
        labels = [-100] * len(ids)
        # MLM: mask 15% of tokens
        # Only mask content positions (not [CLS] or [SEP])
        for pos in range(1, len(ids) - 1):
            if self.rng.random() < self.mask_prob:
                labels[pos] = ids[pos]
                r = self.rng.random()
                if r < 0.8:      # 80% -> [MASK]
                    ids[pos] = self.mask
                elif r < 0.9:    # 10% -> random token
                    ids[pos] = self.rng.randrange(self.vocab_size)
                # 10% -> unchanged
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

