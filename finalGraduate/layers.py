import torch
import torch.nn as nn
import torch.nn.functional as nn_func

from quantizators import quantize_tensor


'''
Класс, реализующий хранение типа квантизации, 
количество прогнанных батчей и данные для расчета статической квантизации
Функционал:
 get_vectors - возвращает вектор в зависимости от типа квантизации
 train_quant - высчитывает максимальный модуль батча
 set_quantize - меняет квантизируемость, тип и битность
'''


class Quantizer(nn.Module):
    def __init__(self, quantize: bool = False, bitness: int = 0):

        super().__init__()
        self.quantize = quantize
        self.bitness = bitness
        self.n_quant = 0
        self.max_inp = []
        self.max_weigh = []
        self.quantize_type = ""
        self.vector_quantizer = quantize_tensor

        self.max_inp_v = torch.nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.max_weigh_v = torch.nn.Parameter(torch.tensor([0.]), requires_grad=True)

        self.register_parameter(name='max_inp_vv',
                                param=self.max_inp_v)
        self.register_parameter(name='max_weigh_vv',
                                param=self.max_weigh_v)

    def forward(self, input_tensor, weight):
        # print(input_tensor.requires_grad, weight.requires_grad)

        if self.quantize:
            if self.quantize_type == "dynamic":
                input_v = self.vector_quantizer(input_tensor, self.bitness)
                weights_v = self.vector_quantizer(weight, self.bitness)
            else:
                input_v = self.vector_quantizer(input_tensor, self.bitness, self.max_inp_v[0])
                weights_v = self.vector_quantizer(weight, self.bitness, self.max_weigh_v[0])
        else:
            input_v = input_tensor
            weights_v = weight

        return input_v, weights_v

    def train_quant(self, input_tensor, weight):
        self.max_inp.append(torch.max(torch.abs(input_tensor)))
        self.max_weigh.append(torch.max(torch.abs(weight)))

        self.n_quant += 1

        with torch.no_grad():
            self.max_inp_v[0] = sum(self.max_inp) / len(self.max_inp)
            self.max_weigh_v[0] = sum(self.max_weigh) / len(self.max_weigh)

    def set_quantize(self, quantize, bitness, quantize_type, trainable):
        self.quantize = quantize
        self.bitness = bitness
        self.quantize_type = quantize_type

        self.max_inp_v.requires_grad = trainable
        self.max_weigh_v.requires_grad = trainable


'''
Класс, реализующий интерфейс квантизованного слоя, связанный с Quantizer
Функционал:
 train_quant - высчитывает максимальный модуль батча
 set_quantize - меняет квантизируемость, тип и битность
'''


class QuantedLayer:
    def __init__(self):
        self.quant = None

    def set_quantize(self, quantize: bool, bitness: int, quantize_type: str, trainable):
        self.quant.set_quantize(quantize, bitness, quantize_type, trainable)

    def train_quant(self, input_tensor):
        res = self.forward(input_tensor)

        self.quant.train_quant(input_tensor, self.weight)

        return res

    def get_scalerss(self):
        # print(self.quant.max_inp_v.requires_grad, self.quant.max_weigh_v.requires_grad, self.weight.requires_grad)
        # print(self.quant.max_inp_v.grad_fn, self.quant.max_weigh_v.grad_fn, self.weight.grad_fn)

        print(self.quant.max_inp_v.data, self.quant.max_weigh_v.data)
        # return self.quant.max_inp_v, self.quant.max_weigh_v


'''
Переписанные слои
'''

def check_gradient_fn(t, s):
    if t.grad_fn is not None and s[0] != "i":
        print(s + " " + str(t.grad_fn))
    return t


class Conv2d(nn.Conv2d, QuantedLayer):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: tuple = (4, 4), stride: int = 1,
                 pad_dint: int = 0, bias: bool = True,
                 quantize: bool = False, bitness: int = 0):
        super(Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=pad_dint, bias=bias)

        self.quant = Quantizer(quantize, bitness)

    def forward(self, input_tensor):
        input_v, weights_v = self.quant.forward(input_tensor, self.weight)

        ret = nn_func.conv2d(input_v, weights_v,
                              self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        return ret


class Linear(nn.Linear, QuantedLayer):
    def __init__(self,
                 in_features: int, out_features: int,
                 bias: bool = False,
                 quantize: bool = False, bitness: int = 0):
        super(Linear, self).__init__(in_features, out_features, bias)

        self.quant = Quantizer(quantize, bitness)

    def forward(self, input_tensor):
        # check_gradient_fn(input_tensor, "ilb")
        # check_gradient_fn(self.weight, "wlb")

        input_v, weights_v = self.quant.forward(input_tensor, self.weight)
        ret = nn_func.linear(input_v, weights_v, self.bias)

        # check_gradient_fn(input_tensor, "ila")
        # check_gradient_fn(self.weight, "wla")

        return ret
