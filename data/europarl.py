"""Europarl text corpus loader with simple tokenizer."""

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class SimpleTokenizer:
    """Word-level tokenizer with special tokens."""

    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    def build_vocab(self, sentences):
        """Build vocabulary from list of sentences."""
        counter = Counter()
        for sent in sentences:
            counter.update(sent.lower().split())
        for word, _ in counter.most_common(self.vocab_size - 4):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, sentence: str, max_len: int = 30):
        """Encode sentence to token indices."""
        words = sentence.lower().split()
        tokens = [self.SOS]
        for w in words[:max_len - 2]:
            tokens.append(self.word2idx.get(w, self.UNK))
        tokens.append(self.EOS)
        # Pad
        while len(tokens) < max_len:
            tokens.append(self.PAD)
        return tokens

    def decode(self, tokens):
        """Decode token indices to sentence."""
        words = []
        for t in tokens:
            if t == self.EOS:
                break
            if t in (self.PAD, self.SOS):
                continue
            words.append(self.idx2word.get(t, "<UNK>"))
        return " ".join(words)


class EuroparlDataset(Dataset):
    """Europarl parallel corpus (English) for text semantic communication."""

    def __init__(self, filepath: str, tokenizer: SimpleTokenizer,
                 max_len: int = 30, max_samples: int = 100000):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentences = []

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Europarl file not found: {filepath}. "
                "Download from https://www.statmt.org/europarl/ "
                "and place the .txt file at the specified path."
            )

        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                line = line.strip()
                if len(line.split()) >= 4:  # Skip very short sentences
                    self.sentences.append(line)

        tokenizer.build_vocab(self.sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.sentences[idx], self.max_len)
        return torch.tensor(tokens, dtype=torch.long)


def get_europarl_loaders(filepath: str, batch_size: int = 64,
                         vocab_size: int = 10000, max_len: int = 30,
                         num_workers: int = 2):
    """Get Europarl DataLoader with tokenizer."""
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    dataset = EuroparlDataset(filepath, tokenizer, max_len=max_len)

    # 90/10 split
    n_train = int(0.9 * len(dataset))
    n_test = len(dataset) - n_train
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader, tokenizer
