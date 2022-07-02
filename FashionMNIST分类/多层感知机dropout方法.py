from Config import *
from util import get_test_data, get_train_data, get_fashion_mnist_labels, show_image

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.reshape(-1, input_size)
        out_1 = self.fc1(x)
        self.relu(out_1)
        out_1 = self.dropout(out_1)
        out_2 = self.fc2(out_1)
        self.relu(out_2)
        out_2 = self.dropout(out_2)
        out = self.fc3(out_2)
        return out


if __name__ == '__main__':
    train_dataset = get_train_data()
    test_dataset = get_test_data()
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size, shuffle=False)
    Model = Net()
    optimizer = optim.SGD(Model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter('../runs/fashion_mnist_Multilayer_Dropout')
    step = 0

    for epoch in range(num_epoch):
        Model.train()
        train_accuracy_list = []
        for batch_x, batch_y in tqdm(train_iter):
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

    torch.save(Model.state_dict(), 'MLP_dropout')