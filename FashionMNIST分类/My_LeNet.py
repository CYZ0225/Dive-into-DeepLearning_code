import torch
from torch import nn, optim
from Config import *
from util import show_image, get_fashion_mnist_labels, get_train_data, get_test_data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 这里padding的原因是因为原始的Net输入是32,而我们的输出是28,因此共padding4个全0
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        out = self.conv1(x)
        out = self.sigmoid(out)
        # [batch, 1, 28, 28] -> [batch, 6, 28, 28]
        out = self.pool(out)
        # [batch, 6, 28, 28] -> [batch, 6, 14, 14]
        out = self.conv2(out)
        out = self.sigmoid(out)
        # [batch, 6, 14, 14] -> [batch, 16, 10, 10]
        out = self.pool(out)
        # [batch, 16, 10, 10] -> [batch, 16, 5, 5]
        out = out.reshape(-1, 5*5*16)
        # [batch, 16, 5, 5] -> [batch, 400]
        out = self.fc1(out)
        # [batch, 400] -> [batch, 120]
        out = self.sigmoid(out)
        out = self.fc2(out)
        # [batch, 120] -> [batch, 84]
        out = self.sigmoid(out)
        out = self.fc3(out)
        # [batch, 84] -> [batch, 10]
        return out


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    train_dataset = get_train_data()
    test_dataset = get_test_data()
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size, shuffle=False)
    Model = Net().to(device)
    Model.apply(init_weights)  # 不用这个初始化的话,效果很差
    optimizer = optim.SGD(Model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss().to(device)
    writer = SummaryWriter('../runs/fashion_mnist_LeNet')
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
            accuracy = accuracy_score(batch_y, y_pred.detach().numpy())
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
                batch_x = batch_x.reshape(-1, 784)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = Model(batch_x)
                y_pred = torch.argmax(out, dim=1)
                accuracy = accuracy_score(batch_y, y_pred.detach().numpy())
                test_accuracy_list.append(accuracy)

            test_accuracy_list = np.array(test_accuracy_list)
            test_acc = np.mean(test_accuracy_list)
            tqdm.write(f'epoch:{epoch + 1},test_accuracy{test_acc}')

    eval_iter = DataLoader(train_dataset, batch_size=5, shuffle=True)
    x, y = next(iter(eval_iter))
    Model.eval()
    out = Model(x)
    pred = torch.argmax(out, dim=1)

    pred_titles = get_fashion_mnist_labels(pred)
    true_titles = get_fashion_mnist_labels(y)
    title = ["pred label:" + pred_title + '\n' + 'true label:' + true_title
             for pred_title, true_title in zip(pred_titles, true_titles)]
    show_image(x, title=title)

    writer.close()

    torch.save(Model.state_dict(), 'LeNet')