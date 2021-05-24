import torch

import torch.nn as nn
import torch.nn.functional as nn_func
from typing import Union, List
from torch import Size
import numbers
from torch.nn.parameter import Parameter

from models.CNN_test.quantizators import QuantedLayer, Quantizer

'''
Переписанные слои
'''


class Conv2d(nn.Conv2d, QuantedLayer):
    """
    Overwritten pytorch Conv2d layer with adding quantizing attributes and methods
    """

    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: tuple = (4, 4), stride: int = 1,
                 padding: int = 0, bias: bool = True,
                 quantize: bool = False, bitness: int = 0):
        super(Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, bias=bias)

        self.quant = Quantizer(quantize, bitness)

    def forward(self, input_tensor):
        input_v, weights_v = self.quant.forward(input_tensor, self.weight)
        ret = nn_func.conv2d(input_v, weights_v,
                             self.bias, self.stride,
                             self.padding, self.dilation, self.groups)

        return ret


class Linear(nn.Linear, QuantedLayer):
    """
    Overwritten pytorch Linear layer with adding quantizing attributes and methods
    """

    def __init__(self,
                 in_features: int, out_features: int,
                 bias: bool = False,
                 quantize: bool = False, bitness: int = 0):
        super(Linear, self).__init__(in_features, out_features, bias)

        self.quant = Quantizer(quantize, bitness)

    def forward(self, input_tensor):
        input_v, weights_v = self.quant.forward(input_tensor, self.weight)
        # print("====================")
        # print(f'inp:\n {torch.min(input_v)} \t {torch.max(input_v)} \n '
        #       f'wei:\n {torch.min(weights_v)} \t {torch.max(weights_v)}')
        ret = nn_func.linear(input_v, weights_v, self.bias)
        # print(f'ret: \n {torch.min(ret)} \t  {torch.max(ret)} \n')
        # print(f'bias: \n {self.bias}')

        return ret

