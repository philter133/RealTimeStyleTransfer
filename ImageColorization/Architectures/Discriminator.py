import torch
import torch.nn as nn


def get_block(in_fil: int,
              out_fil: int,
              kernal=(4, 4),
              stride=(2, 2),
              padding=(1, 1),
              norm=True,
              activation=True) -> nn.Sequential:
    layers = [nn.Conv2d(in_fil,
                        out_fil,
                        kernal,
                        stride,
                        padding,
                        bias=not norm)]

    if norm:
        layers.append(nn.BatchNorm2d(out_fil))
    if activation:
        layers.append(nn.LeakyReLU(0.2))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self,
                 num_filters: int,
                 block_count: int):
        super(Discriminator, self).__init__()

        model = [get_block(3, num_filters, norm=False)]

        for i in range(block_count - 1):
            model.append(get_block(num_filters * 2 ** i,
                                   num_filters * 2 ** (i + 1)))
        i += 1

        model.append(get_block(num_filters * 2 ** i,
                               num_filters * 2 ** (i + 1),
                               stride=(1, 1)))

        model.append(get_block(num_filters * 2 ** block_count,
                               1,
                               stride=(1, 1),
                               norm=False,
                               activation=False))

        self.__model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__model(x)


if __name__ == '__main__':
    test = Discriminator(64, 3)

    print(test)


