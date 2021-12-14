import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from Models.Transformer import TransformerNetwork

torch.manual_seed(108)


def get_transformer_list(normalize: bool,
                         range_255: bool) -> list:
    transformation_list = [transforms.Resize(256),
                           transforms.CenterCrop(256),
                           transforms.ToTensor()]
    if range_255:
        transformation_list.append(transforms.Lambda(lambda x: x.mul(255)))
        if normalize:
            transformation_list.append(transforms.Normalize(mean=np.array([123.675, 116.28, 103.53]),
                                                            std=np.array([1, 1, 1])))
    else:
        if normalize:
            transformation_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
    return transformation_list


# trnasforms images for being fed into the model
def transform_image(state_dict_path: str,
                    content_image_path,
                    save_path=None,
                    save=False,
                    normalize=False,
                    range_255=False):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # device_name = "cpu"
    device = torch.device(device=device_name)

    checkpoint = torch.load(state_dict_path)
    model = TransformerNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        model.eval()

        pic_tf_list = get_transformer_list(normalize=normalize,
                                           range_255=range_255)[2:]

        pic_tf = transforms.Compose(pic_tf_list)

        content_image = load_image(content_image_path)
        content_tensor = torch.unsqueeze(pic_tf(content_image), dim=0).to(device)

        display_image(torch.squeeze(model(content_tensor), dim=0).cpu(),
                      range_255=range_255,
                      normalize=normalize,
                      save=save,
                      path=save_path)


def compute_gram(image: torch.Tensor):
    b, c, h, w = image.size()
    image = image.view(b, c, h * w)
    t_image = torch.transpose(image, 1, 2)
    gram_matrix = torch.bmm(image, t_image) / (c * h * w)
    return gram_matrix


def display_image(data,
                  save=False,
                  path=None,
                  normalize=False,
                  range_255=False):
    img = data.detach().clone().numpy()

    if range_255:
        if normalize:
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([1, 1, 1])
            img = (img * std + mean).transpose(1, 2, 0).clip(0, 255).astype("uint8")
        else:
            img = img.transpose(1, 2, 0).clip(0, 255).astype("uint8")

    else:
        if normalize:
            std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
            mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
            img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")

        else:
            img = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")

    img = Image.fromarray(img)
    img.show()

    if save:
        img.save(path)


def load_image(filename):
    img = Image.open(filename).convert('RGB')
    return img
