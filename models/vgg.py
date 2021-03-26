import math

from torch.nn.parameter import Parameter
import torch.nn as nn
from models.module import Conv2d, Linear, Dropout, MaxPool2d, BatchNorm2d
import torch
from models.activate_fn import ReLU


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EmbeddingVGG(nn.Module):

    def __init__(self, features):
        super(EmbeddingVGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        self.embedding = Parameter(torch.ones(1, 3, 32, 32))  # the simplest way to embedding image

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x * self.embedding)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


def Vgg16():
    return VGG(make_layers(cfg['D']))


def Vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))


def EmbeddingVgg16():
    return EmbeddingVGG(make_layers(cfg['D']))


def EmbeddingVgg16_bn():
    return EmbeddingVGG(make_layers(cfg['D'], batch_norm=True))


class AbpVgg16(nn.Module):
    def __init__(self):
        super(AbpVgg16, self).__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8 = Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv9 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv10 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv11 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv13 = Conv2d(512, 512, kernel_size=3, padding=1)

        self.linear1 = Linear(512, 512)
        self.linear2 = Linear(512, 512)
        self.linear3 = Linear(512, 10)

        self.dropout1 = Dropout()
        self.dropout2 = Dropout()

        self.relu = ReLU()

        self.pooling = MaxPool2d(2)

        self.save_for_another_backward = None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, Linear):
            #     m.linear.weight.data.normal_(0, 0.01)
            #     m.linear.bias.data.zero_()

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
        x15 = self.conv7(x14)
        x16 = self.relu(x15)
        x17 = self.pooling(x16)
        x18 = self.conv8(x17)
        x19 = self.relu(x18)
        x20 = self.conv9(x19)
        x21 = self.relu(x20)
        x22 = self.conv10(x21)
        x23 = self.relu(x22)
        x24 = self.pooling(x23)
        x25 = self.conv11(x24)
        x26 = self.relu(x25)
        x27 = self.conv12(x26)
        x28 = self.relu(x27)
        x29 = self.conv13(x28)
        x30 = self.relu(x29)
        x31 = self.pooling(x30)
        x32 = x31.view(-1, 512)

        x33 = self.dropout1(x32)
        x34 = self.linear1(x33)  # x33
        x35 = self.relu(x34)
        x36 = self.dropout2(x35)
        x37 = self.linear2(x36)  # x36
        x38 = self.relu(x37)
        x39 = self.linear3(x38)

        if self.training:
            x1.retain_grad()
            x3.retain_grad()
            # x4.retain_grad()
            x6.retain_grad()
            x8.retain_grad()
            # x9.retain_grad()
            x11.retain_grad()
            x13.retain_grad()
            x15.retain_grad()
            # x16.retain_grad()
            x18.retain_grad()
            x20.retain_grad()
            x22.retain_grad()
            # x23.retain_grad()
            x25.retain_grad()
            x27.retain_grad()
            x29.retain_grad()
            # x30.retain_grad()
            x34.retain_grad()
            x37.retain_grad()
            x39.retain_grad()
            if self.save_for_another_backward is not None:
                print('Warning: save_for_another_backward is overwritten.')
            self.save_for_another_backward = [
                x1, x3, x4, x6, x8, x9, x11, x13, x15, x16, x18, x20, x22, x23, x25, x27, x29, x30, x34, x37, x39
            ]

        return x39

    def another_backward(self, another_grad):
        x1, x3, x4, x6, x8, x9, x11, x13, x15, x16, x18, x20, x22, x23, x25, x27, x29, x30, x34, x37, x39 = \
            self.save_for_another_backward
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
            grad15 = self.conv7.another_backward(grad14, x15.grad)
            grad16 = self.relu.another_backward(grad15, x15)
            grad17 = self.pooling.another_backward(grad16, x16)
            grad18 = self.conv8.another_backward(grad17, x18.grad)
            grad19 = self.relu.another_backward(grad18, x18)
            grad20 = self.conv9.another_backward(grad19, x20.grad)
            grad21 = self.relu.another_backward(grad20, x20)
            grad22 = self.conv10.another_backward(grad21, x22.grad)
            grad23 = self.relu.another_backward(grad22, x22)
            grad24 = self.pooling.another_backward(grad23, x23)
            grad25 = self.conv11.another_backward(grad24, x25.grad)
            grad26 = self.relu.another_backward(grad25, x25)
            grad27 = self.conv12.another_backward(grad26, x27.grad)
            grad28 = self.relu.another_backward(grad27, x27)
            grad29 = self.conv13.another_backward(grad28, x29.grad)
            grad30 = self.relu.another_backward(grad29, x29)
            grad31 = self.pooling.another_backward(grad30, x30)
            grad32 = grad31.view(-1, 512)

            grad33 = self.dropout1.another_backward(grad32)
            grad34 = self.linear1.another_backward(grad33, x34.grad)
            grad35 = self.relu.another_backward(grad34, x34)
            grad36 = self.dropout2.another_backward(grad35)
            grad37 = self.linear2.another_backward(grad36, x37.grad)
            grad38 = self.relu.another_backward(grad37, x37)
            ______ = self.linear3.another_backward(grad38, x39.grad)

        self.save_for_another_backward = None

    def update_grad(self, weight):
        with torch.no_grad():
            self.conv1.update_grad(weight)
            self.conv2.update_grad(weight)
            self.conv3.update_grad(weight)
            self.conv4.update_grad(weight)
            self.conv5.update_grad(weight)
            self.conv6.update_grad(weight)
            self.conv7.update_grad(weight)
            self.conv8.update_grad(weight)
            self.conv9.update_grad(weight)
            self.conv10.update_grad(weight)
            self.conv11.update_grad(weight)
            self.conv12.update_grad(weight)
            self.conv13.update_grad(weight)
            self.linear1.update_grad(weight)
            self.linear2.update_grad(weight)
            self.linear3.update_grad(weight)


class AbpVgg16bn(nn.Module):
    def __init__(self):
        super(AbpVgg16bn, self).__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(64)
        self.conv2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(64)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2d(128)
        self.conv4 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = BatchNorm2d(128)
        self.conv5 = Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = BatchNorm2d(256)
        self.conv6 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = BatchNorm2d(256)
        self.conv7 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = BatchNorm2d(256)
        self.conv8 = Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = BatchNorm2d(512)
        self.conv9 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = BatchNorm2d(512)
        self.conv10 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = BatchNorm2d(512)
        self.conv11 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn11 = BatchNorm2d(512)
        self.conv12 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12 = BatchNorm2d(512)
        self.conv13 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn13 = BatchNorm2d(512)

        self.linear1 = Linear(512, 512)
        self.linear2 = Linear(512, 512)
        self.linear3 = Linear(512, 10)

        self.dropout1 = Dropout()
        self.dropout2 = Dropout()

        self.relu = ReLU()

        self.pooling = MaxPool2d(2)

        self.save_for_another_backward = None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, Linear):
            #     m.linear.weight.data.normal_(0, 0.01)
            #     m.linear.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        x6 = self.relu(x5)
        x7 = self.pooling(x6)
        x8 = self.conv3(x7)
        x9 = self.bn3(x8)
        x10 = self.relu(x9)
        x11 = self.conv4(x10)
        x12 = self.bn4(x11)
        x13 = self.relu(x12)
        x14 = self.pooling(x13)
        x15 = self.conv5(x14)
        x16 = self.bn5(x15)
        x17 = self.relu(x16)
        x18 = self.conv6(x17)
        x19 = self.bn6(x18)
        x20 = self.relu(x19)
        x21 = self.conv7(x20)
        x22 = self.bn7(x21)
        x23 = self.relu(x22)
        x24 = self.pooling(x23)
        x25 = self.conv8(x24)
        x26 = self.bn8(x25)
        x27 = self.relu(x26)
        x28 = self.conv9(x27)
        x29 = self.bn9(x28)
        x30 = self.relu(x29)
        x31 = self.conv10(x30)
        x32 = self.bn10(x31)
        x33 = self.relu(x32)
        x34 = self.pooling(x33)
        x35 = self.conv11(x34)
        x36 = self.bn11(x35)
        x37 = self.relu(x36)
        x38 = self.conv12(x37)
        x39 = self.bn12(x38)
        x40 = self.relu(x39)
        x41 = self.conv13(x40)
        x42 = self.bn13(x41)
        x43 = self.relu(x42)
        x44 = self.pooling(x43)
        x45 = x44.view(-1, 512)

        x46 = self.dropout1(x45)
        x47 = self.linear1(x46)  # x33
        x48 = self.relu(x47)
        x49 = self.dropout2(x48)
        x50 = self.linear2(x49)  # x36
        x51 = self.relu(x50)
        x52 = self.linear3(x51)

        if self.training:
            x1.retain_grad()
            x2.retain_grad()
            x4.retain_grad()
            x5.retain_grad()
            x8.retain_grad()
            x9.retain_grad()
            x11.retain_grad()
            x12.retain_grad()
            x15.retain_grad()
            x16.retain_grad()
            x18.retain_grad()
            x19.retain_grad()
            x21.retain_grad()
            x22.retain_grad()
            x25.retain_grad()
            x26.retain_grad()
            x28.retain_grad()
            x29.retain_grad()
            x31.retain_grad()
            x32.retain_grad()
            x35.retain_grad()
            x36.retain_grad()
            x38.retain_grad()
            x39.retain_grad()
            x41.retain_grad()
            x42.retain_grad()
            x47.retain_grad()
            x50.retain_grad()
            x52.retain_grad()
            if self.save_for_another_backward is not None:
                print('Warning: save_for_another_backward is overwritten.')
            self.save_for_another_backward = [
                x1, x2, x4, x5, x6, x8, x9, x11, x12, x13, x15, x16, x18, x19, x21, x22, x23, x25, x26, x28, x29, x31,
                x32, x33, x35, x36, x38, x39, x41, x42, x43, x47, x50, x52
            ]

        return x52

    def another_backward(self, another_grad):
        x1, x2, x4, x5, x6, x8, x9, x11, x12, x13, x15, x16, x18, x19, x21, x22, x23, x25, x26, x28, x29, x31, \
            x32, x33, x35, x36, x38, x39, x41, x42, x43, x47, x50, x52 = self.save_for_another_backward
        with torch.no_grad():
            grad1 = self.conv1.another_backward(another_grad, x1.grad)
            grad2 = self.bn1.another_backward(grad1, x2.grad)
            grad3 = self.relu.another_backward(grad2, x2)
            grad4 = self.conv2.another_backward(grad3, x4.grad)
            grad5 = self.bn2.another_backward(grad4, x5.grad)
            grad6 = self.relu.another_backward(grad5, x5)
            grad7 = self.pooling.another_backward(grad6, x6)
            grad8 = self.conv3.another_backward(grad7, x8.grad)
            grad9 = self.bn3.another_backward(grad8, x9.grad)
            grad10 = self.relu.another_backward(grad9, x9)
            grad11 = self.conv4.another_backward(grad10, x11.grad)
            grad12 = self.bn4.another_backward(grad11, x12.grad)
            grad13 = self.relu.another_backward(grad12, x12)
            grad14 = self.pooling.another_backward(grad13, x13)
            grad15 = self.conv5.another_backward(grad14, x15.grad)
            grad16 = self.bn5.another_backward(grad15, x16.grad)
            grad17 = self.relu.another_backward(grad16, x16)
            grad18 = self.conv6.another_backward(grad17, x18.grad)
            grad19 = self.bn6.another_backward(grad18, x19.grad)
            grad20 = self.relu.another_backward(grad19, x19)
            grad21 = self.conv7.another_backward(grad20, x21.grad)
            grad22 = self.bn7.another_backward(grad21, x22.grad)
            grad23 = self.relu.another_backward(grad22, x22)
            grad24 = self.pooling.another_backward(grad23, x23)
            grad25 = self.conv8.another_backward(grad24, x25.grad)
            grad26 = self.bn8.another_backward(grad25, x26.grad)
            grad27 = self.relu.another_backward(grad26, x26)
            grad28 = self.conv9.another_backward(grad27, x28.grad)
            grad29 = self.bn9.another_backward(grad28, x29.grad)
            grad30 = self.relu.another_backward(grad29, x29)
            grad31 = self.conv10.another_backward(grad30, x31.grad)
            grad32 = self.bn10.another_backward(grad31, x32.grad)
            grad33 = self.relu.another_backward(grad32, x32)
            grad34 = self.pooling.another_backward(grad33, x33)
            grad35 = self.conv11.another_backward(grad34, x35.grad)
            grad36 = self.bn11.another_backward(grad35, x36.grad)
            grad37 = self.relu.another_backward(grad36, x36)
            grad38 = self.conv12.another_backward(grad37, x38.grad)
            grad39 = self.bn12.another_backward(grad38, x39.grad)
            grad40 = self.relu.another_backward(grad39, x39)
            grad41 = self.conv13.another_backward(grad40, x41.grad)
            grad42 = self.bn13.another_backward(grad41, x42.grad)
            grad43 = self.relu.another_backward(grad42, x42)
            grad44 = self.pooling.another_backward(grad43, x43)
            grad45 = grad44.view(-1, 512)

            grad46 = self.dropout1.another_backward(grad45)
            grad47 = self.linear1.another_backward(grad46, x47.grad)
            grad48 = self.relu.another_backward(grad47, x47)
            grad49 = self.dropout2.another_backward(grad48)
            grad50 = self.linear2.another_backward(grad49, x50.grad)
            grad51 = self.relu.another_backward(grad50, x50)
            ______ = self.linear3.another_backward(grad51, x52.grad)

        self.save_for_another_backward = None

    def update_grad(self, weight):
        with torch.no_grad():
            self.conv1.update_grad(weight)
            self.bn1.update_grad()
            self.conv2.update_grad(weight)
            self.bn2.update_grad()
            self.conv3.update_grad(weight)
            self.bn3.update_grad()
            self.conv4.update_grad(weight)
            self.bn4.update_grad()
            self.conv5.update_grad(weight)
            self.bn5.update_grad()
            self.conv6.update_grad(weight)
            self.bn6.update_grad()
            self.conv7.update_grad(weight)
            self.bn7.update_grad()
            self.conv8.update_grad(weight)
            self.bn8.update_grad()
            self.conv9.update_grad(weight)
            self.bn9.update_grad()
            self.conv10.update_grad(weight)
            self.bn10.update_grad()
            self.conv11.update_grad(weight)
            self.bn11.update_grad()
            self.conv12.update_grad(weight)
            self.bn12.update_grad()
            self.conv13.update_grad(weight)
            self.bn13.update_grad()
            self.linear1.update_grad(weight)
            self.linear2.update_grad(weight)
            self.linear3.update_grad(weight)
