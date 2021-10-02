import torch
import numpy as np

from PIL import Image


def compute_gram(image: torch.Tensor):
    b, c, h, w = image.size()

    image = image.view(b, c, h * w)

    t_image = torch.transpose(image, 1, 2)

    gram_matrix = torch.bmm(image, t_image) / (c * h * w)

    return gram_matrix


def view_image(data):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.detach().clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)

    img.show()


def save_image(data, path):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.detach().clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)

    img.save(path)


def load_image(filename):
    img = Image.open(filename).convert('RGB')
    return img
