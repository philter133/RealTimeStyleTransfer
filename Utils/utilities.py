import torch


def compute_gram(image: torch.Tensor):
    b, c, h, w = image.size()

    image = image.view(b, c, h * w)

    t_image = torch.transpose(image, 1, 2)

    gram_matrix = torch.bmm(image, t_image) / (c * h * w)

    return gram_matrix
