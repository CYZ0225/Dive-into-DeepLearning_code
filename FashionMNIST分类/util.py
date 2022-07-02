import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True,
                                                transform=trans,
                                                download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False,
                                               transform=trans,
                                               download=True)


def get_train_data():
    return mnist_train


def get_test_data():
    return mnist_test


def show_image(imgs, title=None):
    if imgs.dim() == 4:
        plt.figure(figsize=(4 * imgs.shape[0], 4))
        for i, img in enumerate(imgs):
            if img.shape == (1, 28, 28):
                img = img.reshape(28, 28, 1)
            plt.subplot(1, imgs.shape[0], i+1)
            plt.title(title[i])
            plt.imshow(img)
        plt.show()

    if imgs.dim() == 3:
        if imgs.shape == (1, 28, 28):
            imgs = imgs.reshape(28, 28, 1)

            plt.imshow(imgs)
            plt.title(title)
            plt.show()


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker','bag','ankle boot']
    if torch.is_tensor(labels):
        return [text_labels[int(i)] for i in labels]
    elif type(labels) == int:
        return text_labels[labels]


if __name__ == "__main__":
    print(len(mnist_train))  # 60000
    print(len(mnist_test))  # 10000

    # 这里提取第一个样本对,里面包含了样本和标签
    print(mnist_train[0][0].shape) # [1, 28, 28]
    # x = mnist_train[3][0]
    # y = mnist_train[3][1]
    #
    # labels = get_fashion_mnist_labels(y)
    # show_image(x, labels)
    # print(labels)
    dataloader = DataLoader(mnist_train, batch_size=5)
    for batch_x, batch_y in dataloader:
        print(batch_x.shape[0])
        labels = get_fashion_mnist_labels(batch_y)
        show_image(batch_x, title=labels)
        break