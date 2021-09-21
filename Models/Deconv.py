import torch
import torch.nn as nn


class Deconvolution(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernal_size: tuple,
                 stride: int,
                 output_padding: int):
        super(Deconvolution, self).__init__()

        self.__activation = nn.ReLU()

        self.__normalization = nn.BatchNorm2d(out_channel)

        self.__conv_layer = nn.ConvTranspose2d(in_channel,
                                               out_channel,
                                               kernal_size,
                                               stride=stride,
                                               padding=kernal_size[0] // 2,
                                               output_padding=output_padding)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = self.__conv_layer(in_tensor)
        out_tensor = self.__normalization(out_tensor)

        return self.__activation(out_tensor)
