import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        self.__features = models.vgg16(pretrained=True).features

        self.__relu_1_2 = nn.Sequential()
        self.__relu_2_2 = nn.Sequential()
        self.__relu_3_3 = nn.Sequential()
        self.__relu_4_3 = nn.Sequential()

        for x in range(23):
            if x < 4:
                self.__relu_1_2.add_module(str(x), self.__features[x])
            elif 4 <= x < 9:
                self.__relu_2_2.add_module(str(x), self.__features[x])
            elif 9 <= x < 16:
                self.__relu_3_3.add_module(str(x), self.__features[x])
            else:
                self.__relu_4_3.add_module(str(x), self.__features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image_tensor: torch.Tensor):
        relu_dict = {}

        output = self.__relu_1_2(image_tensor)
        relu_dict["relu1_2"] = output

        output = self.__relu_2_2(output)
        relu_dict["relu2_2"] = output

        output = self.__relu_3_3(output)
        relu_dict["relu3_3"] = output

        output = self.__relu_4_3(output)
        relu_dict["relu4_3"] = output

        return relu_dict
