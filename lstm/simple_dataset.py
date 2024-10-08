import torch
from torch.utils.data import Dataset
import numpy as np


class SimpleDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        print(self.data.shape)
        self.labels = np.load(label_path)
        self.size = self.data.shape[0]
        assert (
            self.data.shape[0] == self.labels.shape[0]
        ), "Data and labels must have the same number of samples"
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.int64)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


if __name__ == "__main__":
    dataset = SimpleDataset(
        "data/data.npy", "data/labels.npy"
    )  # Paths to the data and labels files
    for i in range(len(dataset)):
        x, y = dataset[i]  # Directly call the __getitem__ method using indexing
    print(x.__len__())
    print(x.shape, y.shape)   # torch.Size([128, 2]) torch.Size([])
    print("------------")
    print((dataset[0][0]).shape)  # torch.Size([])
    print(dataset[0][1].shape)  # tensor(0)
    print(x.size(), y.size())
