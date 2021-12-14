import torch
import torch.nn as nn

# A deconvoloutional block in the model architecture

class Deconvolution(nn.Module):

    # def __init__(self,
    #              in_channel: int,
    #              out_channel: int,
    #              kernal_size: tuple,
    #              stride: int,
    #              output_padding: int):
    #     super(Deconvolution, self).__init__()
    #
    #     self.__activation = nn.ReLU()
    #
    #     # self.__normalization = nn.BatchNorm2d(out_channel)
    #     self.__normalization = nn.InstanceNorm2d(out_channel, affine=True)
    #
    #     self.__conv_layer = nn.ConvTranspose2d(in_channel,
    #                                            out_channel,
    #                                            kernal_size,
    #                                            stride=stride,
    #                                            padding=kernal_size[0] // 2,
    #                                            output_padding=output_padding)
    #
    # def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
    #     out_tensor = self.__conv_layer(in_tensor)
    #     out_tensor = self.__normalization(out_tensor)
    #
    #     return self.__activation(out_tensor)

    """UpsampleConvLayer
        Upsamples the input and then does a convolution. This method gives better results
        compared to ConvTranspose2d.
        ref: http://distill.pub/2016/deconv-checkerboard/
     """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(Deconvolution, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.__normalization = nn.InstanceNorm2d(out_channels, affine=True)
        self.__activation = nn.ReLU()

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return self.__activation(self.__normalization(out))
