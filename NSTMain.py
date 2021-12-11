import io
import typing
import torch.nn.functional as F
import numpy as np
import torch
from typing import Union
from PIL import Image
from torchvision import transforms
from VGG import VGGModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(img: typing.Union[bytes, str],
               size: int):
    transformer_list = [transforms.Resize((size, size), transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(size),
                        transforms.ToTensor()]

    composer = transforms.Compose(transformer_list)

    if isinstance(img, str):
        image = Image.open(img)
    else:
        image = Image.open(io.BytesIO(img))

    image_arr = torch.unsqueeze(composer(image), dim=0).to(device, dtype=torch.float)

    return image_arr


def show_image(img: torch.Tensor):
    image = img.cpu().clone()
    image = torch.squeeze(image, dim=0)
    arr = image.numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    im = Image.fromarray(arr)
    im.show()
    return im


def get_loss(ground_truth: typing.List[torch.Tensor],
             layer_list: typing.List[torch.Tensor],
             weights: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(ground_truth[0], layer_list[0])

    for idx, layer in enumerate(layer_list[1:]):
        loss += F.mse_loss(ground_truth[idx + 1], layer)

    return loss


def gram_matrix(matrix: torch.Tensor):
    (batch, channels, rows, columns) = matrix.size()

    matrix = matrix.view(batch, channels, rows * columns)

    transposed_matrix = torch.transpose(matrix, 1, 2)

    gram = torch.bmm(matrix, transposed_matrix)

    return gram / (channels * rows * columns)


def normalize_images(tensor: torch.Tensor,
                     mean: torch.Tensor,
                     std: torch.Tensor):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    return (tensor - mean) / std


def denormalize_images(tensor,
                       mean,
                       std):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    return (tensor * std) + mean


def white_noise_image(size):
    return torch.from_numpy(np.random.rand(3, size, size))


class NeuralStyleTransfer:

    def __init__(self,
                 style_image: Union[bytes, str],
                 content_image: Union[bytes, str],
                 image_size: int,
                 layer_set: str,
                 avg_pool: bool,
                 content_input: bool,
                 layer_weights: typing.List[float],
                 content_weight: float,
                 style_weight: float,
                 lr=1e1):
        self.__style_weight = style_weight
        self.__content_weight = content_weight

        style = load_image(style_image,
                           image_size)

        content = load_image(content_image,
                             image_size)

        conv_dict = {"a": {"content": [(1, 1)], "style": [(1, 1)]},
                     "b": {"content": [(2, 1)], "style": [(1, 1), (2, 1)]},
                     "c": {"content": [(3, 1)], "style": [(1, 1), (2, 1), (3, 1)]},
                     "d": {"content": [(4, 1)], "style": [(1, 1), (2, 1), (3, 1), (4, 1)]},
                     "e": {"content": [(5, 1)], "style": [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]}}

        self.__vgg_model = VGGModel(conv_dict[layer_set]["style"],
                                    conv_dict[layer_set]["content"],
                                    device,
                                    avg_pool=avg_pool).eval()

        self.__mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.__std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        style = normalize_images(style,
                                 self.__mean,
                                 self.__std)

        content = normalize_images(content,
                                   self.__mean,
                                   self.__std)

        _, content_gt = self.__vgg_model(content)
        self.__content_gt = [x.detach() for x in content_gt]

        style_gt, _ = self.__vgg_model(style)
        self.__style_gt_list = [gram_matrix(x).detach() for x in style_gt]

        self.__input = white_noise_image(image_size) if not content_input else content.clone()
        self.__input.requires_grad_(True)

        self.__style_layer_weight = torch.tensor(layer_weights,
                                                 dtype=torch.float64,
                                                 device=device)

        self.__content_layer_weight = torch.tensor(layer_weights[:1],
                                                   dtype=torch.float64,
                                                   device=device)

        self.__optimizer = torch.optim.Adam([self.__input], lr)

    def train_one_adam(self, steps):
        for i in range(steps):
            self.__optimizer.zero_grad()

            style_output, content_output = self.__vgg_model(self.__input)

            content_loss = get_loss(self.__content_gt,
                                    content_output,
                                    self.__content_layer_weight)

            gram_style_list = [gram_matrix(x) for x in style_output]

            style_loss = get_loss(self.__style_gt_list,
                                  gram_style_list,
                                  self.__style_layer_weight)

            full_loss = (content_loss * self.__content_weight) + (style_loss * self.__style_weight)

            full_loss.backward()

            self.__optimizer.step()

            return full_loss

        # with torch.no_grad():
        #     # temp = torch.clamp(self.__input, 0, 1)
        #     temp = denormalize_images(self.__input, self.__mean, self.__std)
        #     show_image(temp)

    def get_image(self):

        with torch.no_grad():
            # temp = torch.clamp(self.__input, 0, 1)
            temp = denormalize_images(self.__input, self.__mean, self.__std)
            show_image(temp)
            return show_image(temp)



if __name__ == '__main__':
    nst = NeuralStyleTransfer("D:/Postman/women.jpeg",
                              "D:/Postman/me.jpg",
                              500,
                              "e",
                              True,
                              True,
                              [1, 1, 1, 1, 1],
                              1e-2,
                              1e5,
                              0.01)

    for i in range(100):
        nst.train_one_adam(100)
