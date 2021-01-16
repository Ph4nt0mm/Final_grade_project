import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Conv2d, Linear

'''
CNN model.

'''


# from torch.autograd import Variable
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 n_filters, filter_sizes,
                 output_dim, dropout,
                 quantize=False, bitness=0):
        super().__init__()

        self.quantize = quantize

        ''' Layers part '''

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            Conv2d(in_channels=1, out_channels=n_filters,
                   kernel_size=(fs, embedding_dim), quantize=quantize,
                   bitness=bitness)
            for fs in filter_sizes
        ])

        self.fc = Linear(len(filter_sizes) * n_filters, output_dim, quantize=quantize, bitness=bitness)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        input_tensor = input_tensor.permute(1, 0)

        embedded = self.embedding(input_tensor)

        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)

    def train_quant(self, input_tensor):
        input_tensor = input_tensor.permute(1, 0)

        embedded = self.embedding(input_tensor)

        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv.train_quant(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc.train_quant(cat)

    def set_quantize_layers(self, quantize: bool = False, bitness: int = 0, quantize_type: str = "dynamic", trainable=False):
        for i in self.modules():
            try:
                i.set_quantize(quantize, bitness, quantize_type, trainable)
            except:
                continue

    def get_scalers(self):
        for i in self.modules():
            try:
                i.get_scalerss()
            except:
                continue
