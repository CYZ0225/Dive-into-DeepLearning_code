from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset

from Config import *
from 线性回归 import make_data


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=True)
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    x, y = make_data(True_w,True_b,sample_size)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    Model = Net()
    optimizer = optim.SGD(Model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            output = Model(batch_x)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch:{epoch+1},loss:{loss},W:{Model.linear.weight},b:{Model.linear.bias}')
