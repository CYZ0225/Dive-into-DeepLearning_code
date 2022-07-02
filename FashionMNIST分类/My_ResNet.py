import torch
from torch import nn, optim
from torchsummary import summary
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torchvision
from torchvision import transforms
from Config import *


class Residual(nn.Module):
    """
    实现一个残差连接网络的一个block, 不会改变输入的形状

    use_1x1conv是残差连接时,是否要做一次卷积(1X1卷积可以认为是一个特殊的全连接层)

    Note:
         input_channels == num_channerls -> use_1X1conv 都可以运行
         input_channels != num_channerls -> use_1X1conv 必须等于True
         这里是因为残差连接 Y和 X shape 的不一致
    """
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.use_1x1conv = use_1x1conv

        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # conv3是1x1卷积,就是残差连接的时候要不要做一个全连接
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        Y = self.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))

        if self.use_1x1conv:
            x = self.conv3(x)
        Y += x
        return self.relu(Y)


class Residual_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block_2 = nn.Sequential(Residual(64, 64),
                                     Residual(64, 64, use_1x1conv=True, strides=2))
        self.block_3 = nn.Sequential(Residual(64, 128, use_1x1conv=True, strides=2),
                                     Residual(128, 128))
        self.block_4 = nn.Sequential(Residual(128, 256, use_1x1conv=True, strides=2),
                                     Residual(256, 256))
        self.block_5 = nn.Sequential(Residual(256, 512, use_1x1conv=True, strides=2),
                                     Residual(512, 512))
        self.block_6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Flatten(),
                                     nn.Linear(512, 10))

    def forward(self, x):
        # x.shape = [batch, 1, 96, 96]
        out = self.block_1(x)
        # [batch, 1, 96, 96] -> [batch, 64, 48, 48]
        out = self.block_2(out)
        # [batch, 64, 48, 48] -> [batch, 64, 24, 24]
        out = self.block_3(out)
        # [batch, 64, 24, 24] -> [batch, 128, 12, 12]
        out = self.block_4(out)
        # [batch, 128, 12, 12] -> [batch, 256, 6, 6]
        out = self.block_5(out)
        # [batch, 256, 6, 6] -> [batch, 512, 3, 3]
        out = self.block_6(out)
        # [batch, 512, 3, 3] -> [batch, 10]
        return out


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    # # 打印网络层
    # Model = Residual_Net()
    # summary(Model, (1, 96, 96))

    trans = transforms.Compose([transforms.Resize((96, 96)),
                                transforms.ToTensor()])
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False,
                                                   transform=trans,
                                                   download=True)

    train_iter = DataLoader(mnist_train, batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False)
    Model = Residual_Net().to(device)
    Model.apply(init_weights)  # 不用这个初始化的话,效果很差
    optimizer = optim.SGD(Model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    writer = SummaryWriter('../runs/fashion_mnist_ResNet')
    step = 0

    for epoch in range(num_epoch):
        Model.train()
        train_accuracy_list = []
        for batch_x, batch_y in tqdm(train_iter):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = Model(batch_x)
            loss = criterion(out, batch_y)
            writer.add_scalar('loss', loss, step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(out, dim=1)
            accuracy = accuracy_score(batch_y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            train_accuracy_list.append(accuracy)
            writer.add_scalar('train_accuracy', accuracy, step)

            step += 1

        train_accuracy_list = np.array(train_accuracy_list)
        avg_acc = train_accuracy_list.mean()
        tqdm.write(f'epoch:{epoch + 1}, train_accuracy:{avg_acc}')

        with torch.no_grad():
            Model.eval()
            test_accuracy_list = []
            for batch_x, batch_y in test_iter:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = Model(batch_x)
                y_pred = torch.argmax(out, dim=1)
                accuracy = accuracy_score(batch_y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
                test_accuracy_list.append(accuracy)

            test_accuracy_list = np.array(test_accuracy_list)
            test_acc = np.mean(test_accuracy_list)
            tqdm.write(f'epoch:{epoch + 1},test_accuracy{test_acc}')


    writer.close()

    torch.save(Model.state_dict(), 'ResNet')