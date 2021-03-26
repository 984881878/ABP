import torch.nn as nn


class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU(inplace)
        # self.x = None

    def forward(self, x):
        # if self.relu.inplace:
        #     self.x = x.detach().clone()
        # else:
        #     self.x = x.detach()
        return self.relu(x)

    @staticmethod
    def another_backward(another_grad, x):
        mask = (x.data > 0)
        grad = mask * another_grad.data
        # self.x = None  # clear tmp data manually after get grad, release reference of it.
        return grad


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.y = None

    def forward(self, x):
        # self.y = self.sigmoid(x)
        return self.sigmoid(x)

    @staticmethod
    def another_backward(another_grad, y):
        grad = y.data * (1 - y.data) * another_grad.data
        # self.y = None   # clear tmp data manually after get grad, release reference of it
        return grad
