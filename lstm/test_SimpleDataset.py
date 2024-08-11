from simple_dataset import SimpleDataset


train_dataset = SimpleDataset(data_path='data/data.npy', label_path='data/labels.npy')
print(train_dataset[0][0].size(), train_dataset[0][1].size())
#print(train_dataset[0])

#print(train_dataset[127])
print(train_dataset[10000])
print(train_dataset[1][0])
print(train_dataset[1][1])
# print(train_dataset[0].size())  'tuple' object has no attribute 'size'
