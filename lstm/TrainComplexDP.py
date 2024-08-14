# @file trainpipeline.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# This is the dataset we built
from ComplexModel import ComplexModel
from ComplexDataset import ComplexDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"!!! Device: {DEVICE}")
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01
N_SEQ = 32
SEQ_LEN = 128
HIDDEN_DIM = 256


def train():
    # Create a model and move it to DEVICE
    model = ComplexModel(n_seq=N_SEQ, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM).to(DEVICE)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    print(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    print("model been parallelized")
    model = nn.DataParallel(model)  # this is dp model
    model.to(device)


    # Create train dataset and dataloader
    train_dataset = ComplexDataset(data_path='data/complex-data.npy', label_path='data/complex-labels.npy')
    val_dataset = ComplexDataset(data_path='data/complex-data.npy', label_path='data/complex-labels.npy')
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # Create a loss function and an optimizer; The optimizer will update the model's parameters
    criterion = nn.CrossEntropyLoss()  # since it is a 6-class classification problem
    # optimizer should be `Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print("Model output contains NaN or Inf values!")
            print(y_pred.shape)
            assert y.min() >= 0 and y.max() < 6, "标签值超出范围"
            loss = criterion(y_pred, y.squeeze())  # 去除张量中所有维度为 1 的维度
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
