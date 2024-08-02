import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler  # 分发
from torch.nn.parallel import DistributedDataParallel as DDP  # 封装了 DataParallel
from torch.distributed import (
    init_process_group,
    destroy_process_group,
)  # init 一个world，destroy 一个world

# run.bash 就分配不同rank，回到33.py, 得到分布式训练

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
        model: torch.nn.Module,
        gpu_id: int,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
    ) -> None:
        # rank
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        # 用DDP把模型封装起来
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, xs, ys):
        self.optimizer.zero_grad()
        output = self.model(xs)
        loss = F.cross_entropy(output, ys)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
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
            self._run_batch(xs, ys)

    def train(self, max_epoch: int):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)


# pipeline
class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


# 样本量2048，batchsize 32，那么一个 epoch 会有 2048/32=64 个 batch, 所以每个epoch需要step 64次
train_dataset = MyTrainDataset(2048)


def main(rank: int, world_size: int, max_epochs: int, batch_size: int):

    train_dataset = MyTrainDataset(2048)
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
    model = torch.nn.Linear(20, 1)
    optimzer = torch.optim.SGD(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model, gpu_id=rank, optimizer=optimzer, train_dataloader=train_dataloader
    )
    trainer.train(max_epochs)

    destroy_process_group()


# 分布式training
if __name__ == "__main__":

    # 不同于spaw，从这里已经开始多进程了。因为是run.bash开始的
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


    # main(rank=rank, world_size, max_epochs, batch) 不行，因为开始指定参数，后面就都要指定。

    main(max_epochs=args.max_epochs, batch_size=args.batch_size, rank=rank, world_size=world_size)
