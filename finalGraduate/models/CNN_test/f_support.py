# import matplotlib.pyplot as plt
import numpy as np
import torch

#
# def my_plotter(data: np.array, names: np.array) -> None:
#     """
#     :param data: array of float with sizes N_groups X N_lines X N_epochs
#     :param names: array of str with sizes N_groups withe names of groups
#     :return: None
#     """
#
#     plt.figure()
#
#     for name, groups in zip(names, data):
#         for line in groups:
#             plt.plot(line[0], label=name + " loss")
#             plt.plot(line[1], label=name + " acc")
#     plt.legend()
#     plt.show()


def fix_min_max(first, second):
    """
    Get two values and return in order from min two max
    """

    min_v = min(first, second)
    max_v = max(first, second)

    return min_v, max_v


def get_in_segment(inp: torch.tensor, min: float, max: float) -> np.array:
    """
    :param inp: array of values
    :param min: min available value
    :param max: max available value
    :return: array of values not greater than max and lower than min with saving base order
    """

    min, max = fix_min_max(min, max)

    # res = inp[inp < max]
    # res = res[res > min]

    return inp[torch.logical_and(inp > min, inp < max)]


class MyRound(torch.autograd.Function):
    """
    default round function available to use in pytorch forward|backward module functions
    """

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output