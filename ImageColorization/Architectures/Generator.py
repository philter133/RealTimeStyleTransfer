import torch
import torch.nn as nn


class UnetDown(nn.Module):

    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 dropout: float,
                 norm=True,
                 bias=True) -> None:
        super(UnetDown, self).__init__()

        self.__norm = norm
        self.__conv = nn.Conv2d(in_filters,
                                out_filters,
                                (4, 4),
                                (2, 2),
                                (1, 1),
                                bias=bias)

        # Maybe change to Batch Normalization
        self.__norm = nn.InstanceNorm2d(out_filters)

        self.__activation = nn.LeakyReLU(0.2, True)

        self.__drop = nn.Dropout(p=dropout)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        output = self.__conv(x)
        if self.__norm:
            output = self.__norm(output)

        return self.__drop(self.__activation(output))


class UnetUp(nn.Module):

    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 dropout: float,
                 bias=True) -> None:
        super(UnetUp, self).__init__()

        self.__conv = nn.ConvTranspose2d(in_filters,
                                         out_filters,
                                         (4, 4),
                                         (2, 2),
                                         (1, 1),
                                         bias=bias)

        # Maybe change to Batch Normalization
        self.__norm = nn.InstanceNorm2d(out_filters)

        self.__activation = nn.ReLU(True)

        self.__drop = nn.Dropout(p=dropout)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        output = self.__norm(self.__conv(x))
        return self.__drop(self.__activation(output))


class Unet(nn.Module):

    def __init__(self,
                 down_conv: int,
                 in_channels: int,
                 out_channels: int,
                 in_size: int) -> None:

        super(Unet, self).__init__()

        self.__down_list = []

        for i in range(down_conv - 1):
            layer



