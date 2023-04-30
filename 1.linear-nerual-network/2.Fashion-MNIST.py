import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
import sys

# import d2l
file_dir = os.path.dirname(__file__)
d2l_dir = os.path.join(file_dir, '..')
sys.path.append(d2l_dir)
import d2l.torch as d2l

# get dataset
mnist_train = torchvision.datasets.FashionMNIST(
    root='~/Datasets/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                               train=False,
                                               download=True,
                                               transform=transforms.ToTensor())

# they have __getitem__ and __len__ methods implemented.
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width


def get_fashion_mnist_labels(labels):
    """Convert numerical labels into corresponding text labels."""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    """Draw multiple images and corresponding labels in one line"""
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])

# show_fashion_mnist(X, get_fashion_mnist_labels(y))

# read mini batches
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0 means that no additional processes are used to speed up reading data
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers)

# start = time.time()
# for X, y in train_iter:
#     continue
# print(f'{(time.time() - start):.2f} sec')

####################

# # final function
# def load_data_fashion_mnist(batch_size, resize=None):
#     """Download the Fashion-MNIST dataset and then load it into memory.

#     Defined in :numref:`sec_fashion_mnist`"""
#     # 1. load dataset
#     trans = [transforms.ToTensor()]
#     if resize:
#         trans.insert(0, transforms.Resize(resize))
#     trans = transforms.Compose(trans)
#     mnist_train = torchvision.datasets.FashionMNIST(root="../data",
#                                                     train=True,
#                                                     transform=trans,
#                                                     download=True)
#     mnist_test = torchvision.datasets.FashionMNIST(root="../data",
#                                                    train=False,
#                                                    transform=trans,
#                                                    download=True)

#     # 2. read mini batches
#     return (torch.utils.data.DataLoader(mnist_train,
#                                         batch_size,
#                                         shuffle=True,
#                                         num_workers=num_workers),
#             torch.utils.data.DataLoader(mnist_test,
#                                         batch_size,
#                                         shuffle=False,
#                                         num_workers=num_workers))
