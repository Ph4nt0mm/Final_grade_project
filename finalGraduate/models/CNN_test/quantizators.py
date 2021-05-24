from math import pi

import torch
import numpy as np
from torch import Tensor, sin, nn


from models.CNN_test.f_support import MyRound, get_in_segment


def qsin_sum(input: Tensor, num_bits=2) -> float:
    """
    Calculate qsin loss value for tensor

    :param input: data to quantize
    :param num_bits: degree of int value
    :return: sum of all values in tensor after qsin
    """

    min = -1.0 * (2 ** num_bits) / 2
    max = -min - 1

    res = 0

    pi = torch.tensor(np.pi)

    vec = get_in_segment(input, min, max)

    res = res + torch.sum(torch.square(torch.sin(pi * vec)))

    vec = get_in_segment(input, float(torch.min(input).data)-1, min)
    res = res + torch.sum(torch.square(pi) * torch.square(vec - min))

    vec = get_in_segment(input, max, float(torch.max(input))+1)
    res = res + torch.sum(torch.square(pi) * torch.square(vec + max))

    return res


def quantize_tensor(input_tensor, num_bits, max_v=None) -> Tensor:
    """
    Calculating pseudo quantization of tensor

    :param max_v: in case of dynamic quantization it calculates in function
    otherwise it must be calculated on model level
    :return:
    """

    q_min = -1.0 * (2 ** num_bits) / 2
    q_max = -q_min - 1

    rounder = MyRound.apply

    if max_v is None:
        max_v = torch.max(torch.abs(input_tensor))
    scale = max_v / ((q_max - q_min) / 2)

    input_tensor = torch.div(input_tensor, scale)
    input_tensor = rounder(input_tensor)
    input_tensor = torch.clamp(input_tensor, q_min, q_max)
    # print(input_tensor)
    input_tensor = torch.mul(input_tensor, scale)
    # print(scale)
    # print(input_tensor)
    #
    # print("~~~~~~~~~~")

    # print(input_tensor)

    return input_tensor


class Quantizer(nn.Module):
    """
    Class to calculate quantizing loss and qantized inputs and weights of layers
    """

    def __init__(self, quantize: bool = False, bitness: int = 0):
        """
        init part of module for calculating final view of tensors
        :param quantize: defined if quantizing of layer
        :param bitness: defined degree of int greed

        max_inp - updateable values of activations of layers
        max_weigh - updateable values of weights of layers

        max_inp_v - trainable attribute of activations of layers
        max_weigh_v - trainable attribute of weights of layers

        self.register_parameter just gives names to attributes of layer
        """
        super().__init__()
        self.quantize = quantize
        self.bitness = bitness
        self.n_quant = 0

        self.max_inp = []
        self.max_weigh = []

        self.quantize_type = ""

        self.max_inp_v = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.max_weigh_v = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)

        self.register_parameter(name='max_inp_vv',
                                param=self.max_inp_v)
        self.register_parameter(name='max_weigh_vv',
                                param=self.max_weigh_v)
        self.use_qloss = False

        self.q_w_sum = 0
        self.q_i_sum = 0

    def forward(self, input: torch.tensor, weight: torch.tensor):
        """
        :param input: activations of layers
        :param weight:  weights of layers
        :return: quantized input and weight
        """
        if self.quantize:
            if self.quantize_type == "dynamic":
                input_v = quantize_tensor(input, self.bitness)
                weights_v = quantize_tensor(weight, self.bitness)
            elif self.quantize_type == "static_train":
                self.train_quant(input, weight)

                input_v = input
                weights_v = weight
            else:
                input_v = quantize_tensor(input, self.bitness, self.max_inp_v[0])
                weights_v = quantize_tensor(weight, self.bitness, self.max_weigh_v[0])

                if self.use_qloss:
                    self.q_w_sum = qsin_sum(weight / self.max_weigh_v)
                    self.q_i_sum = qsin_sum(input / self.max_inp_v)

        else:
            input_v = input
            weights_v = weight

        return input_v, weights_v

    def train_quant(self, input_tensor, weight):
        """
        Updates max values of activations and weights
        """

        self.max_inp.append(torch.max(torch.abs(input_tensor)).detach().cpu())
        self.max_weigh.append(torch.max(torch.abs(weight)).detach().cpu())

        with torch.no_grad():
            self.max_inp_v[0] = sum(self.max_inp) / len(self.max_inp)
            self.max_weigh_v[0] = sum(self.max_weigh) / len(self.max_weigh)

    def set_quantize(self, quantize: bool = False, bitness: int = 0,
                            quantize_type: str = "dynamic", trainable=False, use_qloss=False):
        """
        Setting new parameters of quantization for layer
        """

        self.quantize = quantize
        self.bitness = bitness
        self.quantize_type = quantize_type

        self.max_inp_v.requires_grad = trainable
        self.max_weigh_v.requires_grad = trainable

        self.use_qloss = use_qloss


class QuantedLayer:
    """
    Defending functions to inherit in layers to use in higher levels
    """

    def __init__(self):
        self.quant = None

    def set_quantize(self, quantize: bool = False, bitness: int = 0,
                     quantize_type: str = "dynamic",
                     trainable: bool=False, use_qloss: bool=False):
        """
        Setting new parameters of quantization for layer
        """

        self.quant.set_quantize(quantize, bitness, quantize_type, trainable, use_qloss)

    def get_w_los(self):
        """
        :return: normalized qantized weights and inputs
        """

        res = 0

        try:
            inp, res = self.quant.forward(0 * self.weight, self.weight)
        except:
            pass

        return torch.norm(res - self.weight)

    def get_qsin_weights(self):
        """
        :return: qsin loss for weights
        """

        return qsin_sum(self.weight / self.quant.max_weigh_v)

    def get_qsin_inputs(self, tens):
        """
        :return: qsin loss for input tensor
        """

        return qsin_sum(tens / self.quant.max_inp_v)