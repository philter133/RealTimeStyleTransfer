import os
import typing
import numpy as np
import torch

from skimage import color
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

np.random.seed(108)

# lab format to rgb
def lab_to_rgb(L: torch.Tensor,
               ab: torch.Tensor) -> np.ndarray:
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=0)

    Lab = Lab.permute(1, 2, 0).cpu().numpy()

    img_rgb = color.lab2rgb(Lab)

    return (img_rgb * 255).astype(np.uint8)

# dataloader
class ColorizationDataset(Dataset):

    def __init__(self,
                 path: str,
                 data_amount: typing.Optional[int],
                 img_tf: transforms.Compose) -> None:
        self.__data_list = np.array([os.path.join(path, i) for i in os.listdir(path)])
        np.random.shuffle(self.__data_list)

        if data_amount is not None:
            self.__data_list = self.__data_list[:data_amount]

        self.__composer = img_tf

    def __len__(self):
        return len(self.__data_list)

    def __getitem__(self, idx) -> typing.Tuple[torch.Tensor,
                                               torch.Tensor]:
        img = Image.open(self.__data_list[idx]).convert("RGB")
        img = self.__composer(img)
        arr = np.array(img)
        lab = color.rgb2lab(arr).astype("float32")
        ten_lab = transforms.ToTensor()(lab)

        L = ten_lab[0:1] / 50 - 1
        ab = ten_lab[1:] / 110

        return L, ab
