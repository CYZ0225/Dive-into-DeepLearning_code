import torch
import numpy as np
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)


batch_size = 256
lr = 0.1
num_epoch = 10
LR = 1  # 这个学习率是My_LeNet.py中使用的

# softmax 超参数
num_inpus = 1*28*28
num_outputs = 10

# 多层感知机参数据
input_size, hidden_size, output_size = 784, 256, 10

# Dropout
p = 0.2
