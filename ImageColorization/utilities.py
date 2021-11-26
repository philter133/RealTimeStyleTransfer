import os
import typing
import numpy as np
import torch

from skimage import io, color
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

np.random.seed(108)


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

        L = ten_lab[0:1] / 50 + 1
        ab = ten_lab[1:] / 110

        return L, ab


if __name__ == '__main__':
    size = 256
    batch_size = 16

    transformer_list = [transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(size)]

    val_ds = ColorizationDataset("../Data/Coco/2014/val2014",
                                 None,
                                 transforms.Compose(transformer_list))

    test_ds = ColorizationDataset("../Data/Coco/2014/test2014",
                                  None,
                                  transforms.Compose(transformer_list))

    transformer_list.append(transforms.RandomHorizontalFlip())

    train_ds = ColorizationDataset("../Data/Coco/2014/Train/train2014",
                                   None,
                                   transforms.Compose(transformer_list))

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(train_ds,
                             batch_size=batch_size,
                             shuffle=True)

    val_loader = DataLoader(train_ds,
                            batch_size=batch_size,
                            shuffle=True)
