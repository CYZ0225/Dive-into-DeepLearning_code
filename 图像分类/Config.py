import torch
import numpy as np

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

batch_size = 256
lr = 0.1
num_epoch = 10

num_inpus = 1*28*28
num_outputs = 10

# 图像分类.py中的参数
w = torch.normal(0, 0.01, size=(num_inpus, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


if __name__ == '__main__':
    print(w.size())
    print(b.size())
    x = torch.normal(0, 1, (256, 784))
    y = x@w+b
    print(y.size())