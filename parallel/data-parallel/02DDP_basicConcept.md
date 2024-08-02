### node, rank, world_size

> 不太严谨的理解
- world，world_size：
    - world：as a group containing all the processes for your distributed training. （这里的进程指的是一个或多个GPU使用一个模型训练的过程，因此不可再分，所以没有线程的概念）
        - 通常，每一个 gpu 代表一个进程（process）
        - world 内的 process 可以彼此通信，所以有 ddp 分布式训练的；
        
- rank
    - rank: is the unique ID given to a process, 进程级别的概念，rank 是为了标识、识别进程，因为进程间（process）需要通信；
    - local rank：is the a unique local ID for processes running in a single node

- node 理解为一个 server（服务器，物理的机器），2个servers（多机，机器之间需要通信）就是2个nodes
    - 比如每个 node/server/machine 各有4张卡（4 gpus），一个 2 个node/server；
    - world_size: 2\*4 == 8
    - ranks: [0, 1, 2, 3, 4, 5, 6, 7]
    - local_rank: [0, 1, 2, 3], [0, 1, 2, 3]