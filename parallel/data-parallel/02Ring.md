## 分布式数据并行 DDP

**1. 模型的model parameter，优化器的optimizer states 每张卡都会拷贝一次 replicas**
**2. batch切成不同小份，分给GPU（因为data输入不同 所以loss 反向传播计算的梯度都不一样） （DistruibutedSampler来分发到不同GPU)**
**一个step结束之后**，所有卡和模型的优化器以及参数状态要**保持一致**
始终保持卡间模型的model parameter，优化器的optimizer states一致性
**这个分布式的同步算法就是
ring all-reduce algorithm**


![8c1d3985571790d470e1d0efa645881f.png](:/c44467031fd4477dbde80035d918612b)
- 如上图所示，ring all-reduce algorithm
    - 首先会将所有的 gpu cards 连成一个 ring
    - 其同步过程，不需要等待所有的卡都计算完一轮梯度，
    - 经过这个同步过程之后，所有的卡的 models/optimizers 就都会保持一致的状态；

### Ring AllReduce algorithm
![433add50b947779815d1e3442fea50f4.png](:/b6bde456241545839d901be372230382)
- 计算和同步的几个过程
    - GPUs 分别计算损失（forward）和梯度（backward）
    - 梯度的聚合( 其他卡的gradient值都来到gpu0 ，蓝色箭头)
    - （模型/优化器）参数的更新及广播（broadcast）；
参数的聚合和更新会不同
![ff1fbb42f7e25783b2008ac54b462f95.png](:/3ff5beefc6f34bf1a250f4407d85e6c9)


 HPC（high performance computing）的基础算法
- Ring 环形拓扑结构
    - 百度提出来的；
    - 环形的，logical 的（逻辑的，非物理的）
    - 两个过程（基于环形逻辑拓扑）
        - scatter-reduce
        - all gather（broadcast）
     
1.  ![b6aae38c9e0ca9548bc997b287f28115.png](:/ae9e9d729713469782184904dd862759)
 - send Data to only one GPU (reducer)
    - 将会成为通信的瓶颈
 2. ![6b063b2cccf6449fd6ec36974b0f4f60.png](:/4452a5cbf9e84a45a9555eefe6af1a95)
sending data花费时间更少

### ddp相关概念


