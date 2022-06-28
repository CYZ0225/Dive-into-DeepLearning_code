import matplotlib.pyplot as plt
import torch
import numpy as np
from Config import *


def make_data(w, b, num_sample):
    # 生成x,y 其中为了防止x之间的自带的相关性,所以采用正态生成而不是用linspace,y里面含有白噪声
    x = np.random.normal(0, 1, num_sample * w.shape[1]).reshape(num_sample, w.shape[1])
    y = (x @ w.T + b + np.random.normal(0, 0.01, (num_sample, 1)))

    # 转变为tensor
    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)

    return x, y


def data_iter(batch_size, feature, labels):
    num_example = feature.shape[0]
    idx = list(range(num_example))
    np.random.shuffle(idx)

    for i in range(0, num_example, batch_size):
        batch_idx = idx[i:min(i+batch_size, num_example)]
        yield feature[batch_idx], labels[batch_idx]


def linreg(x, w, b):
    return x @ w.T + b


def mse(y, y_hat):
    loss = torch.sum((y-y_hat).T @ (y-y_hat))
    return loss/y.shape[0]


def sgd(params:list, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    x, y = make_data(True_w, True_b, sample_size)

    # 画图展示相关性
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(x[:, 0].detach().numpy(), y.detach().numpy())
    plt.title('y对x1的相关图')

    plt.subplot(1, 2, 2)
    plt.scatter(x[:, 1].detach().numpy(), y.detach().numpy())
    plt.title('y对x2的相关图')

    plt.show()

    Net = linreg
    criterion = mse
    for epoch in range(num_epochs):
        for batch_x, batch_y in data_iter(batch_size,x,y):
            loss = criterion(batch_y, Net(batch_x, w, b))
            loss.backward()
            sgd([w, b], lr, batch_size)

        # 每一个epoch打印出来看看
        with torch.no_grad():
            train_loss = criterion(batch_y, Net(batch_x, w, b))
            print(f'epoch:{epoch+1},loss:{train_loss}, w:{w}, b:{b}')