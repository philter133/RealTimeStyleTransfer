import torch
import torch.nn as nn
from Models.Conv import Convolution
from Models.ResBlock import ResidualBlock
from Models.Deconv import Deconvolution


class TransformerNetwork(nn.Module):

    def __init__(self):
        super(TransformerNetwork, self).__init__()

        self.__conv_1 = Convolution(3, 32, (9, 9), 1)
        self.__conv_2 = Convolution(32, 64, (3, 3), 2)
        self.__conv_3 = Convolution(64, 128, (3, 3), 2)

        self.__block1 = ResidualBlock()
        self.__block2 = ResidualBlock()
        self.__block3 = ResidualBlock()
        self.__block4 = ResidualBlock()
        self.__block5 = ResidualBlock()

        self.__deconv1 = Deconvolution(128, 64, 3, 1, upsample=2)
        self.__deconv2 = Deconvolution(64, 32, 3, 1, upsample=2)

        self.__padding_layer = nn.ReflectionPad2d(4)
        self.__final_conv = nn.Conv2d(32, 3, (9, 9), stride=1)
        # self.__activation_tanh = nn.Tanh()

    def forward(self, input_tensor) -> torch.Tensor:

        output_tensor = self.__conv_1(input_tensor)
        output_tensor = self.__conv_2(output_tensor)
        output_tensor = self.__conv_3(output_tensor)

        output_tensor = self.__block1(output_tensor)
        output_tensor = self.__block2(output_tensor)
        output_tensor = self.__block3(output_tensor)
        output_tensor = self.__block4(output_tensor)
        output_tensor = self.__block5(output_tensor)

        output_tensor = self.__deconv1(output_tensor)
        output_tensor = self.__deconv2(output_tensor)

        output_tensor = self.__final_conv(self.__padding_layer(output_tensor))

        # return self.__activation_tanh(output_tensor)

        return output_tensor


