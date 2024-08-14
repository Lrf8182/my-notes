# @file trainpipeline.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# This is the dataset we built
from ComplexModel import ComplexModel
from ComplexDataset import ComplexDataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler  # 分发
from torch.nn.parallel import DistributedDataParallel as DDP  # 封装了 DataParallel
from torch.distributed import (
    init_process_group,
    destroy_process_group,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"!!! Device: {DEVICE}")
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01
N_SEQ = 32
SEQ_LEN = 128
HIDDEN_DIM = 256


def ddp_setup(rank):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # nccl：NVIDIA Collective Communication Library
    # 分布式情况下的，gpus 间通信

    # init_process_group 时，rank 和 world_size 应该是必需的，
    # 除非使用 init_method="env://" 进行环境变量初始化（这种方法通常适用于多机设置）
    init_process_group(backend="nccl", rank=rank, init_method="env://")
    # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: ComplexModel,
        gpu_id: int,
        optimizer: torch.optim.Optimizer,   # torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion: torch.nn.CrossEntropyLoss,
        train_dataloader: DataLoader,
        val_dataloader:DataLoader
    ) -> None:
        # rank
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        #self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # 用DDP把模型封装起来
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    def _run_batch_train(self, xs, ys):
        init_states = (
            torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(self.gpu_id),
            torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(self.gpu_id),
        )
        self.optimizer.zero_grad()

        y_pred = self.model(xs,init_states)
        loss = self.criterion(y_pred, ys.squeeze())  # 去除张量中所有维度为 1 的维度
        loss.backward()
        self.optimizer.step()

    def _run_batch_val(self, epoch,xs, ys,total_loss,total_correct,total_samples):
        init_states = (
                torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(self.gpu_id),
                torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(self.gpu_id),
            )
        y_pred =self.model(xs, init_states)
        loss = self.criterion(y_pred, ys.squeeze())  # 去除大小为1的维度
        total_loss += loss.item()
        total_correct += (y_pred.argmax(dim=1) == ys.squeeze()).sum().item()
        total_samples += ys.size(0)
        print(
            f"Epoch: {epoch}, Loss: {total_loss / total_samples}, Accuracy: {total_correct / total_samples}"
        )


    def _run_epoch(self, epoch):
        self.model.train()  # Set the model to training mode
        batch_size = len(next(iter(self.train_dataloader))[0])
        print(
            f"[GPU: {self.gpu_id}] Epoch: {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_dataloader)} "
        )
        # set the current epoch for the sampler in the train_dataloader.
        self.train_dataloader.sampler.set_epoch(epoch)
        # each gpu only process its own batch
        for xs, ys in self.train_dataloader:
            xs = xs.to(self.gpu_id)
            ys = ys.to(self.gpu_id)
            self._run_batch_train(xs, ys)

        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            total_loss = 0
            total_correct = 0
            total_samples = 0
        self.val_dataloader.sampler.set_epoch(epoch)
        for xs, ys in self.val_dataloader:
            xs = xs.to(self.gpu_id)
            ys = ys.to(self.gpu_id)
            self._run_batch_val(epoch,xs, ys,total_loss,total_correct,total_samples)
    
               
    def train(self, max_epoch: int):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)


def main(rank: int, world_size: int, max_epochs: int, batch_size: int):
    train_dataset = ComplexDataset(data_path='data/complex-data.npy', label_path='data/complex-labels.npy')
    val_dataset = ComplexDataset(data_path='data/complex-data.npy', label_path='data/complex-labels.npy')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # 启用固定内存（页面锁定内存）。启用此参数可以加速 CPU 到 GPU 的数据传输。
        pin_memory=True,
        shuffle=False,
        # batch input: split to each gpus (且没有任何 overlaping samples 各个 gpu 之间)
        # 用于分布式训练的数据采样器， 保每个进程（GPU）都能获得训练数据的唯一子集
        sampler=DistributedSampler(train_dataset),
    )
    val_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(val_dataset),
    )
    model = ComplexModel(n_seq=N_SEQ, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(
        model=model, gpu_id=rank, optimizer=optimizer,criterion=criterion,train_dataloader=train_dataloader,val_dataloader=val_dataloader
    )
    trainer.train(max_epochs)
    destroy_process_group()

# 分布式training
if __name__ == "__main__":

    import argparse
    # python my-notes/parallel/data-parallel/03.py --max_epochs 5 --batch_size 32
    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "--max_epochs", type=int, help="Total number of epochs to run", default=10
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each node/device(default:32)",
    )
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])  #  自动获取的当前是第几个进程
    ddp_setup(rank)
    world_size = torch.distributed.get_world_size()  # 自动检测当前一共有几个进程
    print(f"!!! Rank: {rank} | World Size: {world_size}")
    main(max_epochs=args.max_epochs, batch_size=args.batch_size, rank=rank, world_size=world_size)

