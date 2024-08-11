# @file trainpipeline.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# This is the model we builta
from simpleModel import SimpleModel1

# This is the dataset we built
from simple_dataset import SimpleDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"!!! Device: {DEVICE}")
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.01
N_SEQ = 128
SEQ_LEN = 2
HIDDEN_DIM = 16


def train():
    # Create a model and move it to DEVICE
    model = SimpleModel1(n_seq=N_SEQ, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM).to(DEVICE)

    # Create train dataset and dataloader
    train_dataset = SimpleDataset(data_path='data/data.npy', label_path='data/labels.npy')
    val_dataset = SimpleDataset(data_path='data/data.npy', label_path='data/labels.npy')
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # Create a loss function and an optimizer; The optimizer will update the model's parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            init_states = (
                torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(DEVICE),
                torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(DEVICE),
            )
            optimizer.zero_grad()
            y_pred = model(x, init_states)
            loss = criterion(y_pred, y.squeeze())
            loss.backward()
            optimizer.step()

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            total_loss = 0
            total_correct = 0
            total_samples = 0
            for i, (x, y) in enumerate(val_dataloader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                init_states = (
                    torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(DEVICE),
                    torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(DEVICE),
                )
                y_pred = model(x, init_states)
                loss = criterion(y_pred, y.squeeze())  # 去除大小为1的维度
                total_loss += loss.item()
                total_correct += (y_pred.argmax(dim=1) == y.squeeze()).sum().item()
                total_samples += y.size(0)

            print(
                f"Epoch: {epoch}, Loss: {total_loss / total_samples}, Accuracy: {total_correct / total_samples}"
            )


if __name__ == "__main__":
    train()
