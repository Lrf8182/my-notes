# 梯度同步：在每个进程计算完梯度后，使用高效的通信库（如 NCCL 或 Gloo）来同步所有进程的梯度，
# 然后每个进程独立更新自己的模型参数。

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler    # 分发
from torch.nn.parallel import DistributedDataParallel as DDP   # 封装了 DataParallel
from torch.distributed import init_process_group, destroy_process_group   # init 一个world，destroy 一个world

world_size = torch.cuda.device_count()
print(world_size)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # rank 0 process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # nccl：NVIDIA Collective Communication Library 
    # 分布式情况下的，gpus 间通信
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 train_dataloader: DataLoader, 
                 optimizer: torch.optim.Optimizer, 
                 gpu_id: int) -> None:
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
        # typically returns a tuple (or list) where the first element ([0]) is the batch of input data (features)
        # the second element ([1]) is the batch of target data (labels). By indexing with [0], you are accessing the batch of input data.
        batch_size = len(next(iter(self.train_dataloader))[0])
        print(f'[GPU: {self.gpu_id}] Epoch: {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_dataloader)}')
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
print(train_dataset[0])
print(len(train_dataset))


def main(rank: int, world_size: int, max_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    
    train_dataset = MyTrainDataset(2048)
    train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              # 启用固定内存（页面锁定内存）。启用此参数可以加速 CPU 到 GPU 的数据传输。
                              pin_memory=True, 
                              shuffle=False, 
                              # batch input: split to each gpus (且没有任何 overlaping samples 各个 gpu 之间)
                              # 用于分布式训练的数据采样器， 保每个进程（GPU）都能获得训练数据的唯一子集
                              sampler=DistributedSampler(train_dataset))
    model = torch.nn.Linear(20, 1)
    optimzer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    trainer = Trainer(model=model, gpu_id=rank, optimizer=optimzer, train_dataloader=train_dataloader)
    trainer.train(max_epochs)
    
    destroy_process_group()

# 分布式training
if __name__=="__main__":
    import argparse
    # 创建一个 ArgumentParser 对象，用于从命令行读取参数。description 参数提供了程序的简要描述。
    # python my-notes/parallel/data-parallel/03.py --max_epochs 5 --batch_size 32
    parser=argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--max_epochs',type=int, help="Total number of epochs to run", default=10)
    parser.add_argument('--batch_size', default=32,type=int, help='Input batch size on each node/device(default:32)')
    # 解析命令行参数并将其存储在 args 对象中。
    args=parser.parse_args()


    # 这个是检测gpu的数量，建立在GPU=进程数的假设上的
    world_size=torch.cuda.device_count()
    # 使用 mp.spawn 启动多个进程来进行分布式训练。main 是要在每个进程中运行的函数，
    # args 包含传递给 main 函数的参数元组，nprocs 指定要启动的进程数，这里等于可用的 GPU 数量 world_size。
    # mp.spawn 是 PyTorch 提供的用于启动多进程并行任务的函数，它会为每个进程运行指定的函数，并传递一些参数。
    # main 是这个函数的名字，mp.spawn 会为每个进程调用 main 函数，并传递进程的 rank 作为第一个参数。
    
    # spawn分出很多个进程，每个进程执行不同的rank。  l129以上都是单进程，到l130才开始多进程
    mp.spawn(main,args=(world_size,args.max_epochs,args.batch_size),nprocs=world_size)





















    # [GPU: 3] Epoch: 0 | Batchsize: 32 | Steps: 16
    # [GPU: 2] Epoch: 0 | Batchsize: 32 | Steps: 16
    # [GPU: 0] Epoch: 0 | Batchsize: 32 | Steps: 16
    # [GPU: 0] Epoch: 1 | Batchsize: 32 | Steps: 16
    # [GPU: 3] Epoch: 1 | Batchsize: 32 | Steps: 16
    # [GPU: 2] Epoch: 1 | Batchsize: 32 | Steps: 16
    # [GPU: 1] Epoch: 1 | Batchsize: 32 | Steps: 16
    # [GPU: 0] Epoch: 2 | Batchsize: 32 | Steps: 16
    # [GPU: 3] Epoch: 2 | Batchsize: 32 | Steps: 16
    # [GPU: 2] Epoch: 2 | Batchsize: 32 | Steps: 16
    # [GPU: 1] Epoch: 2 | Batchsize: 32 | Steps: 16
    # [GPU: 0] Epoch: 3 | Batchsize: 32 | Steps: 16
    # [GPU: 3] Epoch: 3 | Batchsize: 32 | Steps: 16
    # [GPU: 1] Epoch: 3 | Batchsize: 32 | Steps: 16
    # [GPU: 2] Epoch: 3 | Batchsize: 32 | Steps: 16
    # [GPU: 0] Epoch: 4 | Batchsize: 32 | Steps: 16
    # [GPU: 3] Epoch: 4 | Batchsize: 32 | Steps: 16
    # [GPU: 1] Epoch: 4 | Batchsize: 32 | Steps: 16
    # [GPU: 2] Epoch: 4 | Batchsize: 32 | Steps: 16

# torchrun运行

 