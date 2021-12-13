import torch
import torch.nn as nn
import numpy as np


class UnetDown(nn.Module):

    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 dropout: float,
                 norm=True,
                 bias=True) -> None:
        super(UnetDown, self).__init__()

        self.__norm = norm
        layers = [nn.Conv2d(in_filters,
                            out_filters,
                            (4, 4),
                            (2, 2),
                            (1, 1),
                            bias=bias)]

        # Maybe change to Batch Normalization
        if norm:
            layers.append(nn.BatchNorm2d(out_filters))

        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Dropout(p=dropout))

        self.__model = nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self.__model(x)


class UnetUp(nn.Module):

    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 dropout: float,
                 bias=True) -> None:
        super(UnetUp, self).__init__()

        layers = [nn.ConvTranspose2d(in_filters,
                                     out_filters,
                                     (4, 4),
                                     (2, 2),
                                     (1, 1),
                                     bias=bias),
                  nn.BatchNorm2d(out_filters),  # Maybe change to Batch Normalization
                  nn.ReLU(),
                  nn.Dropout(p=dropout)]

        self.__model = nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self.__model(x)


"""
Produces Unet model architecture displayed in the Pix2Pix paper.
Input size must be
"""


class Unet(nn.Module):

    def __init__(self,
                 image_size: int,
                 filters: int,
                 drop: float,
                 bias: bool) -> None:

        super(Unet, self).__init__()
        down_count = np.log2(image_size)

        if down_count % 1 != 0:
            raise Exception("Invalid image size")

        down_count = int(down_count)
        down_list = [UnetDown(1,
                              filters,
                              0.0,
                              False,
                              bias)]

        for _ in range(3):
            down_list.append(UnetDown(filters,
                                      filters * 2,
                                      0.0,
                                      True,
                                      bias))

            filters *= 2

        for _ in range(down_count - 5):
            down_list.append(UnetDown(filters,
                                      filters,
                                      0.0,
                                      True,
                                      bias))

        last_layer = nn.Sequential(nn.Conv2d(filters,
                                             filters,
                                             (4, 4),
                                             (2, 2),
                                             (1, 1)),
                                   nn.ReLU())

        down_list.append(last_layer)

        self.__down_list = nn.ModuleList(down_list)

        up_list = [UnetUp(filters,
                          filters,
                          drop,
                          bias)]

        for _ in range(3):
            up_list.append(UnetUp(filters * 2,
                                  filters,
                                  drop,
                                  bias))

        for _ in range(down_count - 5):
            up_list.append(UnetUp(filters * 2,
                                  filters // 2,
                                  0.0,
                                  bias))

            filters = filters // 2

        self.__up_list = nn.ModuleList(up_list)

        self.__final_layer = nn.Sequential(
            nn.ConvTranspose2d(filters * 2,
                               2,
                               (4, 4),
                               (2, 2),
                               (1, 1)),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_output = []

        output = self.__down_list[0](x)
        down_output.append(output)

        for i in range(1, len(self.__down_list)):
            down_output.append(output)
            output = self.__down_list[i](output)

        x = -1
        for j in range(0, len(self.__up_list)):
            output = self.__up_list[j](output)
            output = torch.cat((output, down_output[x]), dim=1)
            x -= 1

        return self.__final_layer(output)


if __name__ == '__main__':
    model = Unet(256, 64, 0.5, True)

    data = torch.randn(1, 1, 256, 256)

    print(model(data).size())