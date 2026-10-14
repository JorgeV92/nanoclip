"""A vocabulary-free UTF-8 byte tokenizer: PAD, BOS, EOS, then 256 byte values."""

import torch


class ByteTokenizer:
    pad_id, bos_id, eos_id = 0, 1, 2
    vocab_size = 259

    def __init__(self, context_length: int = 64):
        if context_length < 2:
            raise ValueError("context_length must leave room for BOS and EOS")
        self.context_length = context_length

    def encode(self, text: str) -> list[int]:
        data = list(text.encode("utf-8")[: self.context_length - 2])
        ids = [self.bos_id, *(b + 3 for b in data), self.eos_id]
        return ids + [self.pad_id] * (self.context_length - len(ids))

    def decode(self, ids: list[int]) -> str:
        data = []
        for token in ids:
            if token == self.eos_id:
                break
            if token >= 3:
                data.append(token - 3)
        return bytes(data).decode("utf-8", errors="replace")

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor([self.encode(text) for text in texts], dtype=torch.long).reshape(
            len(texts), self.context_length
        )
