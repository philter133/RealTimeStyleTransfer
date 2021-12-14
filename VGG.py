import torch
import torch.nn as nn
import torchvision.models as models
import typing


# The VGG network model for losses
class VGGModel(nn.Module):

    def __init__(self,
                 style_conv: typing.List[typing.Tuple[int, int]],
                 content_conv: typing.List[typing.Tuple[int, int]],
                 device: torch.device,
                 avg_pool=False):

        super(VGGModel, self).__init__()

        style_conv = sorted(style_conv, key=lambda x: (x[0], x[1]))
        content_conv = sorted(content_conv, key=lambda x: (x[0], x[1]))

        max_val = max([style_conv[-1], content_conv[-1]], key=lambda x: (x[0], x[1]))

        final_conv = "conv{}_{}".format(max_val[0], max_val[1])

        self.__style_str = {"conv{}_{}".format(i[0], i[1]) for i in style_conv}
        self.__content_str = {"conv{}_{}".format(j[0], j[1]) for j in content_conv}

        self.__module_list = torch.nn.ModuleList()

        feature_layers = models.vgg19(pretrained=True).features.to(device)

        current_chunk = 1
        current_conv = 1
        current_relu = 1

        self.__name_list = []

        for layer in feature_layers.children():
            if isinstance(layer, nn.Conv2d):
                name = 'conv{}_{}'.format(current_chunk, current_conv)
                current_conv += 1
                self.__module_list.add_module(name, layer)

            elif isinstance(layer, nn.ReLU):
                name = 'relu{}_{}'.format(current_chunk, current_relu)
                current_relu += 1
                self.__module_list.add_module(name, nn.ReLU(inplace=False))

            elif isinstance(layer, nn.MaxPool2d):
                if avg_pool:
                    name = 'avgpool{}'.format(current_chunk, current_relu)
                    self.__module_list.add_module(name, nn.AvgPool2d(kernel_size=(2, 2),
                                                                     stride=2,
                                                                     padding=0,
                                                                     ceil_mode=False))
                else:
                    name = 'maxpool{}'.format(current_chunk, current_relu)
                    self.__module_list.add_module(name, layer)

                current_chunk += 1
                current_conv = 1
                current_relu = 1

            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.__name_list.append(name)

            if name == final_conv:
                break

    def forward(self,
                x: torch.Tensor) -> typing.Tuple[typing.List[torch.Tensor],
                                                 typing.List[torch.Tensor]]:

        style_list = []
        content_list = []

        for name, layer in zip(self.__name_list, self.__module_list):

            x = layer(x)

            if name in self.__style_str:
                style_list.append(x)

            if name in self.__content_str:
                content_list.append(x)

        return style_list, content_list


if __name__ == '__main__':
    conv_content = [(4, 1)]
    conv_style = [(1, 1), (2, 1), (4, 1), (3, 1)]

    model = VGGModel(conv_style,
                     conv_content,
                     avg_pool=True).eval()

    test_ten = torch.rand(1, 3, 256, 256)

    print(len(model(test_ten)[0]))
