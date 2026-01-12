# dataset.py
import torch

class StreamingTokenDataset(torch.utils.data.IterableDataset):
    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []

        for sample in self.dataset:
            text = sample["text"]
            ids = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            buffer.extend(ids)

            while len(buffer) >= self.seq_len + 1:
                x = torch.tensor(buffer[: self.seq_len], dtype=torch.long)
                y = torch.tensor(buffer[1 : self.seq_len + 1], dtype=torch.long)
                buffer = buffer[self.seq_len :]
                yield x, y
