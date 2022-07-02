import torchvision
from torchvision import transforms
from Config import *
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=(11, 11), stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        # [batch, 1, 227, 227] -> [batch, 96, 55, 55]
        out = self.pool(out)
        # [batch, 96, 55, 55] -> [batch, 96, 27, 27]
        out = self.relu(self.conv2(out))
        # [batch, 96, 27, 27] -> [batch, 256, 27, 27]
        out = self.pool(out)
        # [batch, 256, 27, 27] -> [batch, 256, 13, 13]
        out = self.relu(self.conv3(out))
        # [batch, 256, 13, 13] -> [batch, 384, 13, 13]
        out = self.relu(self.conv4(out))
        # [batch, 384, 13, 13] -> [batch, 384, 13, 13]
        out = self.relu(self.conv5(out))
        # [batch, 384, 13, 13] -> [batch, 256, 13, 13]
        out = self.pool(out)
        # [batch, 256, 13, 13] -> [batch, 256, 6, 6]
        out = out.reshape(-1, 256*6*6)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    # # 打印模型结构
    # Model = Net()
    # summary(Model, (1, 227, 227))
    # exit()

    trans = transforms.Compose([transforms.Resize((227, 227)),
                                transforms.ToTensor()])
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False,
                                                   transform=trans,
                                                   download=True)

    train_iter = DataLoader(mnist_train, batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False)
    Model = Net().to(device)
    Model.apply(init_weights)  # 不用这个初始化的话,效果很差
    optimizer = optim.SGD(Model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    writer = SummaryWriter('../runs/fashion_mnist_AlexNet')
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

    # torch.save(Model.state_dict(), 'AlexNet')  # 因为这个存储有233MB大小,所以没有存
