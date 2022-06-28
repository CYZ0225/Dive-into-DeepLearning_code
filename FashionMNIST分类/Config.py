import torch
import numpy as np

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

batch_size = 256
lr = 0.1
num_epoch = 10

# softmax 超参数
num_inpus = 1*28*28
num_outputs = 10

# 多层感知机参数据
input_size, hidden_size, output_size = 784, 256, 10
