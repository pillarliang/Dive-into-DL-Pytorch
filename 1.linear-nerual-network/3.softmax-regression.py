import torch
import torch.nn as nn
import os
import sys

file_dir = os.path.dirname(__file__)
d2l_dir = os.path.join(file_dir, '..')
sys.path.append(d2l_dir)

import d2l.torch as d2l

# 1. load and read dataset
batch_size = 256
# each fetch is a batchsize size.
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# the shape of data is [batch_size, 1, 28, 28] --> [Batch x Channel x Height x Width]
# PyTorch does not implicitly adjust the shape of the input.
# We define a flatten layer before the linear layer to adjust the shape of the network inputs

# 2. define model
# the shape of image is 28*28
input_features_num = 28 * 28
# Because fashion-mnist has 10 labels, so we define the number of output features as 10
output_features_num = 10

net = nn.Sequential()
net.add_module('flatten', nn.Flatten())
net.add_module('linear', nn.Linear(input_features_num, output_features_num))

# 3. Initialize the model parameters
# method 1
# nn.init.normal_(net.linear.weight, mean=0, std=0.01)
# nn.init.constant_(net.linear.bias, val=0)


# method 2
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, val=0)


net.apply(init_weights)

# 4. Softmax and cross-entropy loss functions
loss = nn.CrossEntropyLoss()

# 5. Define the optimization algorithm
# Using a mini-batch stochastic gradient descent optimization algorithm with a learning rate of 0.1.
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# 6. Calculate classification accuracy
def evaluate_average_accuracy(data_iter, net):
    # acc_num: the total number of correctly predicted value
    # n: the total number of sample
    acc_num, n = 0.0, 0
    for X, y in data_iter:
        # X: feature of each sample
        # y: true value of each sample
        # The number of elements with dimension 1 is equal to the batch size.
        acc_num += (net(X).argmax(dim=1).type(
            y.dtype) == y).float().sum().item()
        n += y.shape[0]
    return acc_num / n


# # 7. training model
def train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer):
    for epoch in range(num_epochs):
        if isinstance(net, torch.nn.Module):
            net.train()

        train_loss_num, train_accuracy_num, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)  # predict value
            loss_value = loss(y_hat, y)

            if isinstance(optimizer, torch.optim.Optimizer):
                # Using PyTorch in-built optimizer & loss criterion
                # clear accumulated gradients
                optimizer.zero_grad()
                loss_value.mean().backward()
                optimizer.step()
            else:
                # Using custom built optimizer & loss criterion
                loss_value.sum().backward()
                optimizer(X.shape[0])

            train_loss_num += loss_value.sum().item()
            train_accuracy_num += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.numel()

        # compute the average loss of each epoch
        train_loss = train_loss_num / n
        # compute the average accurancy of each epoch
        train_accuracy = train_accuracy_num / n
        # compute the accurancy of each epoch with test data
        test_acc = evaluate_average_accuracy(test_iter, net)

        # log
        print(
            f'epoch: {epoch +1}, train loss:{train_loss:.4f}, train_accuracy:{train_accuracy:.3f}, test accuracy: {test_acc:.3f}'
        )


num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)

# some questions in this code:
# If I don't use d2l.train_ch3(), the workers should be set to 0 when calling d2l.load_data_fashion_mnist().
