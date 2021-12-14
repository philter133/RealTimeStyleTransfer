import torch
import torch.nn as nn

# A Residual Block block in the model architecture

class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.__activation = nn.ReLU()

        # self.__normalization = nn.BatchNorm2d(128, affine=True)

        # self.__normalization_2 = nn.BatchNorm2d(128, affine=True)

        self.__normalization = nn.InstanceNorm2d(128, affine=True)

        self.__normalization_2 = nn.InstanceNorm2d(128, affine=True)

        self.__conv_layer = nn.Conv2d(128,
                                      128,
                                      (3, 3),
                                      stride=1,
                                      padding_mode="reflect",
                                      padding=1)

        self.__conv_layer_2 = nn.Conv2d(128,
                                        128,
                                        (3, 3),
                                        stride=1,
                                        padding_mode="reflect",
                                        padding=1)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = self.__conv_layer(in_tensor)
        out_tensor = self.__normalization(out_tensor)
        out_tensor = self.__activation(out_tensor)
        out_tensor = self.__conv_layer_2(out_tensor)
        out_tensor = self.__normalization_2(out_tensor)

        return out_tensor + in_tensor
