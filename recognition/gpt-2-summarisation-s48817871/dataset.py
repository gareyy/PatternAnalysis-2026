from datasets import load_dataset
from torch.utils.data import Dataset
import torch

# Login using e.g. `huggingface-cli login` to access this dataset

class BioLaySummDataset(Dataset):
    def __init__(self, split: str, tokeniser, fmt: str = "{} TLDR; {}", max_length=512) -> None:
        self.bls = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split=split)
        self.tokeniser = tokeniser
        self.max_length = max_length
        self.fmt = fmt
    def __len__(self):
        return self.bls.num_rows
    def __getitem__(self, index):
        radreport = str(self.bls['radiology_report'][index])
        stupidreport = str(self.bls['layman_report'][index])
        # possible format complex and stupid report here?
        new = self.fmt.format(radreport, stupidreport)
        encoded = self.tokeniser(new, truncation=True, max_length=self.max_length, padding="max_length")
        return {
                "input_ids": torch.tensor(encoded['input_ids']),
                "attention_mask": torch.tensor(encoded['attention_mask']),
                "labels": torch.tensor(encoded['input_ids']),
                }
