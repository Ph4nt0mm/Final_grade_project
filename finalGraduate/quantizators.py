from math import pi

import torch
from torch import Tensor, sin


# def QSin(x, min, max):
#     if min <= x and x <= max:
#         return sin(pi * x)^2
#     elif x < min:
#         return pi ^ 2 * (x - min)^2
#     elif max < x:
#         return pi ^ 2 * (x - min)^2


class MyRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# class QuantizeVector(torch.autograd.Function):
#     def forward(self, input_tensor, num_bits, max_v=None):
#         q_min = -1.0 * (2 ** num_bits) / 2
#         q_max = -q_min - 1
#
#         rounder = MyRound.apply
#
#         if max_v is None:
#             max_v = torch.max(torch.abs(input_tensor))
#         scale = max_v / ((q_max - q_min) / 2)
#
#         input_tensor = torch.div(input_tensor, scale)
#         input_tensor = rounder(input_tensor)
#         input_tensor = torch.clamp(input_tensor, q_min, q_max)
#
#         return torch.mul(input_tensor, scale)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#
#         return grad_output, None, None


def quantize_tensor(input_tensor, num_bits, max_v=None) -> Tensor:
    q_min = -1.0 * (2 ** num_bits) / 2
    q_max = -q_min - 1

    rounder = MyRound.apply

    if max_v is None:
        max_v = torch.max(torch.abs(input_tensor))
    scale = max_v / ((q_max - q_min) / 2)

    input_tensor = torch.div(input_tensor, scale)
    input_tensor = rounder(input_tensor)
    input_tensor = torch.clamp(input_tensor, q_min, q_max)
    input_tensor = torch.mul(input_tensor, scale)

    return input_tensor
