import torch
import torch.nn as nn

# A convoloutional block in the model architecture
class Convolution(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernal_size: tuple,
                 stride: int):
        super(Convolution, self).__init__()

        self.__activation = nn.ReLU()

        # self.__normalization = nn.BatchNorm2d(out_channel)
        self.__normalization = nn.InstanceNorm2d(out_channel, affine=True)
        self.__padding_layer = nn.ReflectionPad2d(kernal_size[0] // 2)

        # self.__conv_layer = nn.Conv2d(in_channel,
        #                               out_channel,
        #                               kernal_size,
        #                               stride=stride,
        #                               padding_mode="reflect",
        #                               padding=kernal_size[0] // 2)

        self.__conv_layer = nn.Conv2d(in_channel,
                                      out_channel,
                                      kernal_size,
                                      stride=stride)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = self.__padding_layer(in_tensor)
        out_tensor = self.__conv_layer(out_tensor)
        out_tensor = self.__normalization(out_tensor)

        return self.__activation(out_tensor)



