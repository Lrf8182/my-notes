import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2
batch_size = 30
data_size = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
print(device)

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        # 100*5
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        # (5, )
        return self.data[index]
    def __len__(self):
        # 100
        return self.len
    

rand_loader=DataLoader(dataset=RandomDataset(input_size,data_size),batch_size=batch_size,shuffle=True)

# get first batch
# iter函数本身不会获取数据。它只是返回一个迭代器，指向数据加载器的起始位置。
# 只有在调用next函数或其他迭代操作（例如在循环中使用迭代器）时，才会实际从数据加载器中获取数据。
next(iter(rand_loader)).shape
print(next(iter(rand_loader)).shape)

class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        # 5 => 2
        # super().__init__()  python3 更简洁，不需要传递参数
        super(Model, self).__init__()   # 调用父类的构造函数来初始化父类的属性（父类是nn.Module）
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())
        return output
    

model = Model(input_size, output_size)


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    print(model)
    print("model been parallelized")
    model = nn.DataParallel(model)



# 模型的参数buffer已经被放到device（0）上了
model.to(device)   
print(model)

# input_var can be on any device, including CPU,  tensor data
a = torch.randn(3, 4)
print('a.is_cuda', a.is_cuda)
# The parallelized module must have its parameters and buffers on device_ids[0] before calling .cuda(), and convert the rest of the module to device_ids[0] afterwards.
b = a.to('cuda:0')    # 返回一个新的tensor，不改变原来的tensor， a还在cpu上
print('a.is_cuda', a.is_cuda)
print('b.is_cuda', b.is_cuda)

# model data
a = Model(3, 4)
# 输出a模型的第一个参数的is_cuda属性
print(next(a.parameters()).is_cuda)
print(next(a.parameters()))
b = a.to('cuda:0')
print(next(a.parameters()).is_cuda)
print(next(b.parameters()).is_cuda)
# a and b point to the same model 


# a = model(3, 4)
# print(next(a.parameters()).is_cuda)
# b = a.to('cuda:0')
# print(next(a.parameters()).is_cuda)
# print(next(b.parameters()).is_cuda)

for data in rand_loader:
    # input_var can be on any device, including CPU
    input = data.to(device)  # shape: [30, 5]
#     input = data
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())