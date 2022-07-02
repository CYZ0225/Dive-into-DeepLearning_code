import torch

from 多层感知机dropout方法 import Net
from Config import *
from util import get_test_data, get_train_data, get_fashion_mnist_labels, show_image
from torch.utils.data import DataLoader

Model = Net()
Model.load_state_dict(torch.load('MLP_dropout'))

train_dataset = get_train_data()
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