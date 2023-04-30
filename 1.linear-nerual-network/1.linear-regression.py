import torch
import torch.utils.data as Data
from torch import nn
from torch.nn import init
import numpy as np
import torch.optim as optim

"1. generating dataset"
num_inputs = 2  # size of each input sample
num_examples = 1000  # the number of sample
true_w = [2, -3.4]  # weight
true_b = 4.2  # bias
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)),
                        dtype=torch.float)
# model
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# add noise
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float)

"2. 读取数据：使用小批量"
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

"3.define a model"
# # method 1
# class LinearNet(nn.Module):
#     def __init__(self, n_feature) -> None:
#         """
#         Args:
#             n_feature: size of each input sample
#         """
#         super().__init__()
#         self.linear = nn.Linear(n_feature, 1)

#     def forward(self, x):
#         y = self.linear(x)
#         return y

# net = LinearNet(num_inputs)
# print(net)
# # LinearNet(
# #   (linear): Linear(in_features=2, out_features=1, bias=True)
# # )
# init.normal_(net.linear.weight, mean=0, std=0.01)
# init.constant_(net.linear.bias, val=0)

# method2
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

"4. 初始化模型参数"
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

"5. 定义损失函数"
loss = nn.MSELoss()

"6. 定义优化算法"
optimizer = optim.SGD(net.parameters(), lr=0.03)
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.subnet1.parameters()}, # lr=0.03
#                 {'params': net.subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)

"7. 训练模型"
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        loss_v = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        loss_v.backward()  # 自动反向传播
        optimizer.step()  # 更新权重
    # print('epoch %d, loss: %f' % (epoch, loss_v.item()))

dense = net[0]
# print(true_w, dense.weight)
# print(true_b, dense.bias)
# [2, -3.4] Parameter containing:
# tensor([[ 1.9988, -3.4015]], requires_grad=True)
# 4.2 Parameter containing:
# tensor([4.1998], requires_grad=True)

"预测"
test_data = torch.tensor([[2.4, 3.1]])
true_result = true_w[0] * test_data[:, 0] + true_w[1] * test_data[:,
                                                                  1] + true_b
test_result = net(test_data)
print(f'true value is {true_result.item()}')
# true value is -1.5399999618530273

print(f'predict is {test_result.item()}')
# predict is -1.5377836227416992
