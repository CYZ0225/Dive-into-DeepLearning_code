import numpy as np
import torch

# 通用的配置文件
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)


# 线性回归.py的配置文件
True_w = np.array([[2, -3.4]])
True_b = np.array([4.2])
sample_size = 1000
batch_size = 10
w = torch.normal(0, 0.01, size=True_w.shape, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)
lr = 0.03
num_epochs = 10

if __name__ == '__main__':
    num_sample = sample_size
    x = np.random.normal(0, 1, num_sample * w.shape[1]).reshape(num_sample, w.shape[1])
    print(x.shape)
    print((x @ True_w.T+ True_b).shape)
    y = (x @ True_w.T + True_b + np.random.normal(0, 0.1, num_sample)).reshape(-1, 1)