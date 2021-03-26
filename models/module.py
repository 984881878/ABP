import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.another_weight_grad = None
        # self.register_buffer('vt_1', torch.zeros_like(self.linear.weight.detach().data), persistent=False)
        # self.register_buffer('another_weight_grad', torch.zeros(out_features, in_features), persistent=True)
        # self.another_weight_grad.zero_()
        # self.register_buffer('another_bias_grad', None, persistent=False)

    def forward(self, x):
        return self.linear(x)

    '''another backward propagation computes gradients using mechanism the same as the first backward used.'''
    def another_backward(self, another_grad, grad):
        # self.another_weight_grad += torch.matmul(grad.data.t(), another_grad.data)
        self.another_weight_grad = torch.matmul(grad.data.t(), another_grad.data)

        # if self.linear.bias is not None:
        #     _grad = torch.addmm(self.linear.bias.grad.data, another_grad.data, self.linear.weight.data.t())
        # else:
        _grad = another_grad.data.matmul(self.linear.weight.data.t())
        return _grad

    def update_grad(self, weight=.9):    # lr, momentum=.0, grad_decay=.0
        # self.vt_1 = momentum * self.vt_1 + (1 - momentum) * self.another_weight_grad.data
        # self.linear.weight.grad.data = self.linear.weight.grad.data - lr * self.vt_1  # this is an error instance!!!
        self.linear.weight.grad.data = weight * self.linear.weight.grad.data + \
                                       (1 - weight) * self.another_weight_grad.data
        self.another_weight_grad = None


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.another_weight_grad = None

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def another_backward(self, another_grad, grad):
        weight = Parameter(torch.zeros_like(self.weight).copy_(self.weight.data))
        with torch.enable_grad():
            another_grad.requires_grad_(False)
            _grad = F.conv2d(another_grad, weight, None, self.stride,
                             self.padding, self.dilation, self.groups)
            out = _grad * grad.data
            out.sum().backward()
        self.another_weight_grad = weight.grad.data.clone()

        # # this weight could contain any random data, just need the same shape as self.weight
        # weight = Parameter(torch.zeros_like(self.weight))
        # with torch.enable_grad():
        #     out = F.conv2d(another_grad, weight, None, self.stride,
        #                    self.padding, self.dilation, self.groups)
        #     out = out * grad.data
        #     out.sum().backward()
        # self.another_weight_grad = weight.grad.data.clone()
        # _grad = F.conv2d(another_grad, self.weight.data, None, self.stride,
        #                  self.padding, self.dilation, self.groups)

        return _grad.data.clone()

    def update_grad(self, weight=.4):
        self.weight.grad.data = weight * self.weight.grad.data + \
                                       (1 - weight) * self.another_weight_grad.data
        self.another_weight_grad = None


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(MaxPool2d, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size, stride, padding, dilation)

    def forward(self, x):
        return self.pooling(x)

    def another_backward(self, another_grad, x):
        xx = torch.zeros_like(x).copy_(x.data).requires_grad_(True)
        # xx.retain_grad()

        with torch.enable_grad():
            y = self.pooling(xx)
            y.sum().backward()

        xx.grad.data = (xx.grad.data > .0).float()

        return self.pooling(xx.grad.data * another_grad).data


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AvgPool2d, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return self.pooling(x)

    def another_backward(self, another_grad):
        return self.pooling(another_grad)


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        assert 0 <= p < 1
        self.p = p
        self.multiplier = 1.0 / (1.0 - p)
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        mask = (torch.zeros_like(x).uniform_(0, 1) > self.p).float()
        self.mask = mask
        return self.multiplier * mask * x

    def another_backward(self, another_grad):
        _grad = self.multiplier * self.mask * another_grad
        self.mask = None
        return _grad


class BatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__()

        self.batch_norm2d = nn.BatchNorm2d(num_features)

        self.another_weight_grad = None

    def forward(self, x):
        return self.batch_norm2d(x)

    def another_backward(self, another_grad, grad):
        with torch.no_grad():
            denominator = torch.sqrt(self.batch_norm2d.running_var + self.batch_norm2d.eps).view(1, -1, 1, 1)
            _grad = another_grad * self.batch_norm2d.weight.data.view(1, -1, 1, 1) / denominator
            # self.another_weight_grad = (another_grad * grad / denominator).sum(dim=(0, 2, 3))
        return _grad

    def update_grad(self, weight=.4):
        pass
        # self.batch_norm2d.weight.grad.data = weight * self.batch_norm2d.weight.grad.data + \
        #                                      (1 - weight) * self.another_weight_grad.data
        # self.another_weight_grad = None
