from datasets import load_dataset
from torch.utils.data import Dataset
import torch

"""NOTE ON TEST SPLIT
In the BioLaySumm dataset, the layman reports for the test data are missing.

I simulate a test split by splitting the validation set (in its original order) at a specific point 
(around 8000), then treat that as the test split.

In training, I can then adjust hyperparameters around the validation split, use it for early stopping, etc.
This prevents me from overfitting (through hyperparameters) on all validation data.
"""
SPLIT_POINT = 0.8

class BioLaySummDataset(Dataset):
    def __init__(self, split: str, tokeniser, fmt: str, max_length=512, num_rows: int | None = None) -> None:
        assert split in ['train', 'validation', 'test'], "Unsupported Split!"
        self.split = split
        self.bls = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track",
            split="validation" if split in ['validation', 'test'] else split)
        self.tokeniser = tokeniser
        self.max_length = max_length
        self.fmt = fmt
        if self.split == "test":
            self.length = int(self.bls.num_rows * (1-SPLIT_POINT))
            self.split_offset = int(self.bls.num_rows * SPLIT_POINT)
        elif self.split == "validation":
            self.split_offset = 0
            self.length = int(self.bls.num_rows * SPLIT_POINT)
        else:
            self.split_offset = 0
            self.length = self.bls.num_rows
        if num_rows:
            assert num_rows < self.length and num_rows > 0, "Number of rows is out of range!"
            self.length = num_rows
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        radreport = str(self.bls['radiology_report'][index+self.split_offset])
        stupidreport = str(self.bls['layman_report'][index+self.split_offset])
        # possible format complex and stupid report here?
        new = self.fmt.format(radreport, stupidreport)
        encoded = self.tokeniser(new, truncation=True, max_length=self.max_length, padding="max_length")
        return {
                "input_ids": torch.tensor(encoded['input_ids']),
                "attention_mask": torch.tensor(encoded['attention_mask']),
                "labels": torch.tensor(encoded['input_ids']),
                }
