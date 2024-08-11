from simple_dataset import SimpleDataset
from torch.utils.data import DataLoader
BATCH_SIZE=32

train_dataset = SimpleDataset(data_path='data/data.npy', label_path='data/labels.npy')
dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

print((next(iter(dataloader))[0]).size())
print((next(iter(dataloader))[1]).size())