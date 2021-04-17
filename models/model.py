import torch
import torch.nn as nn
from models.module import Linear, Conv2d, MaxPool2d, GlobalAvgPool2d, Dropout
from torch.nn.parameter import Parameter
from models.activate_fn import ReLU


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = Linear(784, 512)
        self.linear2 = Linear(512, 10)

        self.relu = ReLU(inplace=False)
        # self.sigmoid = Sigmoid()

        self.save_for_another_backward = None

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.relu(x1)  # relu must set inplace=False, x1 needs to be retained for another backward without varying.
        x3 = self.linear2(x2)
        if self.training:
            x1.retain_grad()
            x3.retain_grad()
            if self.save_for_another_backward is not None:
                print('Warning: save_for_another_backward is overwritten.')
            self.save_for_another_backward = (x1, x3)
        return x3

    def another_backward(self, another_grad):
        x1, x3 = self.save_for_another_backward
        with torch.no_grad():
            grad1 = self.linear1.another_backward(another_grad, x1.grad)
            grad2 = self.relu.another_backward(grad1, x1)
            _____ = self.linear2.another_backward(grad2, x3.grad)
        self.save_for_another_backward = None

    def update_grad(self, weight):
        with torch.no_grad():
            self.linear1.update_grad(weight)
            self.linear2.update_grad(weight)


class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 10)
        self.relu = nn.ReLU(inplace=False)
        # self.sigmoid = nn.Sigmoid()
        self.embedding = Parameter(torch.ones(1, 784))      # the simplest way to embedding image

    def forward(self, x):
        x1 = self.linear1(self.embedding * x)
        x2 = self.relu(x1)
        x3 = self.linear2(x2)
        # x4 = self.sigmoid(x3)
        return x3


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 10)
        self.relu = nn.ReLU(inplace=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.relu(x1)
        x3 = self.linear2(x2)
        # x4 = self.sigmoid(x3)
        return x3


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.pooling = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.pooling(self.conv1(x)))
        x = self.relu(self.pooling(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EmbeddingCnn(nn.Module):
    def __init__(self):
        super(EmbeddingCnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.embedding = Parameter(torch.ones(1, 1, 28, 28))        # the simplest way to embedding image
        self.pooling = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.pooling(self.conv1(x * self.embedding)))
        x = self.relu(self.pooling(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AbpCnn(nn.Module):
    def __init__(self):
        super(AbpCnn, self).__init__()
        self.conv1 = Conv2d(1, 10, kernel_size=5)
        self.conv2 = Conv2d(10, 20, kernel_size=5)
        self.fc1 = Linear(320, 50)
        self.fc2 = Linear(50, 10)
        self.pooling = MaxPool2d(2)
        self.relu = ReLU()

        self.save_for_another_backward = None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pooling(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.pooling(x4)
        x6 = self.relu(x5)
        x7 = x6.view(-1, 320)
        x8 = self.fc1(x7)
        x9 = self.relu(x8)
        x10 = self.fc2(x9)

        if self.training:
            x1.retain_grad()
            x4.retain_grad()
            x8.retain_grad()
            x10.retain_grad()
            if self.save_for_another_backward is not None:
                print('Warning: save_for_another_backward is overwritten.')
            self.save_for_another_backward = [x1, x2, x4, x5, x8, x10]

        return x10

    def another_backward(self, another_grad):
        x1, x2, x4, x5, x8, x10 = self.save_for_another_backward
        with torch.no_grad():
            grad1 = self.conv1.another_backward(another_grad, x1.grad)
            grad2 = self.pooling.another_backward(grad1, x1)
            grad3 = self.relu.another_backward(grad2, x2)
            grad4 = self.conv2.another_backward(grad3, x4.grad)
            grad5 = self.pooling.another_backward(grad4, x4)
            grad6 = self.relu.another_backward(grad5, x5)
            grad7 = grad6.view(-1, 320)
            grad8 = self.fc1.another_backward(grad7, x8.grad)
            grad9 = self.relu.another_backward(grad8, x8)
            _____ = self.fc2.another_backward(grad9, x10.grad)
        self.save_for_another_backward = None

    def update_grad(self, weight):
        with torch.no_grad():
            self.conv1.update_grad(weight)
            self.conv2.update_grad(weight)
            self.fc1.update_grad(weight)
            self.fc2.update_grad(weight)


class Cnn6(nn.Module):
    def __init__(self):
        super(Cnn6, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = Linear(256, 256)
        self.fc2 = Linear(256, 10)

        self.pooling = nn.MaxPool2d(2)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = ReLU()
        self.dropout = Dropout()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu(x1)
        x3 = self.conv2(x2)
        x4 = self.relu(x3)
        x5 = self.pooling(x4)

        x6 = self.conv3(x5)
        x7 = self.relu(x6)
        x8 = self.conv4(x7)
        x9 = self.relu(x8)
        x10 = self.pooling(x9)

        x11 = self.conv5(x10)
        x12 = self.relu(x11)
        x13 = self.conv6(x12)
        x14 = self.relu(x13)
        x15 = self.global_pooling(x14).flatten(1)

        x16 = self.dropout(x15)
        x17 = self.fc1(x16)
        x18 = self.relu(x17)
        x19 = self.fc2(x18)

        return x19


class AbpCnn6(nn.Module):
    def __init__(self):
        super(AbpCnn6, self).__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = Linear(256, 256)
        self.fc2 = Linear(256, 10)

        self.pooling = MaxPool2d(2)
        self.global_pooling = GlobalAvgPool2d()
        self.relu = ReLU()
        self.dropout = Dropout()

        self.save_for_another_backward = None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu(x1)
        x3 = self.conv2(x2)
        x4 = self.relu(x3)
        x5 = self.pooling(x4)

        x6 = self.conv3(x5)
        x7 = self.relu(x6)
        x8 = self.conv4(x7)
        x9 = self.relu(x8)
        x10 = self.pooling(x9)

        x11 = self.conv5(x10)
        x12 = self.relu(x11)
        x13 = self.conv6(x12)
        x14 = self.relu(x13)
        x15 = self.global_pooling(x14)

        x16 = self.dropout(x15)
        x17 = self.fc1(x16)
        x18 = self.relu(x17)
        x19 = self.fc2(x18)

        if self.training:
            x1.retain_grad()
            x3.retain_grad()
            x6.retain_grad()
            x8.retain_grad()
            x11.retain_grad()
            x13.retain_grad()
            x17.retain_grad()
            x19.retain_grad()
            if self.save_for_another_backward is not None:
                print('Warning: save_for_another_backward is overwritten.')
            self.save_for_another_backward = [x1, x2, x3, x4, x6, x8, x9, x11, x13, x15, x17, x19]

        return x19

    def another_backward(self, another_grad):
        x1, x2, x3, x4, x6, x8, x9, x11, x13, x15, x17, x19 = self.save_for_another_backward
        with torch.no_grad():
            grad1 = self.conv1.another_backward(another_grad, x1.grad)
            grad2 = self.relu.another_backward(grad1, x1)
            grad3 = self.conv2.another_backward(grad2, x3.grad)
            grad4 = self.relu.another_backward(grad3, x3)
            grad5 = self.pooling.another_backward(grad4, x4)

            grad6 = self.conv3.another_backward(grad5, x6.grad)
            grad7 = self.relu.another_backward(grad6, x6)
            grad8 = self.conv4.another_backward(grad7, x8.grad)
            grad9 = self.relu.another_backward(grad8, x8)
            grad10 = self.pooling.another_backward(grad9, x9)

            grad11 = self.conv5.another_backward(grad10, x11.grad)
            grad12 = self.relu.another_backward(grad11, x11)
            grad13 = self.conv6.another_backward(grad12, x13.grad)
            grad14 = self.relu.another_backward(grad13, x13)
            grad15 = self.global_pooling.another_backward(grad14)

            grad16 = self.dropout.another_backward(grad15)
            grad17 = self.fc1.another_backward(grad16, x17.grad)
            grad18 = self.relu.another_backward(grad17, x17)
            ______ = self.fc2.another_backward(grad18, x19.grad)
        self.save_for_another_backward = None

    def update_grad(self, weight):
        with torch.no_grad():
            self.conv1.update_grad(weight)
            self.conv2.update_grad(weight)
            self.conv3.update_grad(weight)
            self.conv4.update_grad(weight)
            self.conv5.update_grad(weight)
            self.conv6.update_grad(weight)
            self.fc1.update_grad(weight)
            self.fc2.update_grad(weight)


class AbpCnnTest(nn.Module):
    def __init__(self):
        super(AbpCnnTest, self).__init__()
        self.conv1 = Conv2d(1, 10, kernel_size=3, padding=1)
        self.conv2 = Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = Conv2d(20, 20, kernel_size=3, padding=1)
        self.fc1 = Linear(20, 50)
        self.fc2 = Linear(50, 10)
        self.pooling1 = MaxPool2d(2)
        self.pooling2 = MaxPool2d(7)
        self.relu = ReLU()

        self.save_for_another_backward = None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pooling1(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.pooling1(x4)
        x6 = self.relu(x5)
        x7 = self.conv3(x6)
        x8 = self.pooling2(x7)
        x9 = self.relu(x8)
        x10 = x9.view(-1, 20)
        x11 = self.fc1(x10)
        x12 = self.relu(x11)
        x13 = self.fc2(x12)

        if self.training:
            x1.retain_grad()
            x4.retain_grad()
            x7.retain_grad()
            x11.retain_grad()
            x13.retain_grad()
            if self.save_for_another_backward is not None:
                print('Warning: save_for_another_backward is overwritten.')
            self.save_for_another_backward = [x1, x2, x4, x5, x7, x8, x11, x11, x13]

        return x13

    def another_backward(self, another_grad):
        x1, x2, x4, x5, x7, x8, x11, x11, x13 = self.save_for_another_backward
        with torch.no_grad():
            grad1 = self.conv1.another_backward(another_grad, x1.grad)
            grad2 = self.pooling1.another_backward(grad1, x1)
            grad3 = self.relu.another_backward(grad2, x2)
            grad4 = self.conv2.another_backward(grad3, x4.grad)
            grad5 = self.pooling1.another_backward(grad4, x4)
            grad6 = self.relu.another_backward(grad5, x5)
            grad7 = self.conv3.another_backward(grad6, x7.grad)
            grad8 = self.pooling2.another_backward(grad7, x7)
            grad9 = self.relu.another_backward(grad8, x8)
            grad10 = grad9.view(-1, 20)
            grad11 = self.fc1.another_backward(grad10, x11.grad)
            grad12 = self.relu.another_backward(grad11, x11)
            ______ = self.fc2.another_backward(grad12, x13.grad)
        self.save_for_another_backward = None

    def update_grad(self, weight):
        with torch.no_grad():
            self.conv1.update_grad(weight)
            self.conv2.update_grad(weight)
            self.conv3.update_grad(weight)
            self.fc1.update_grad(weight)
            self.fc2.update_grad(weight)
