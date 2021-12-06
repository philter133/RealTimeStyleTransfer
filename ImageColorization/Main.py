from Architectures.Discriminator import Discriminator
from Architectures.Generator import Unet
from utils import lab_to_rgb
from utils import ColorizationDataset
from torchvision import transforms
from tqdm import tqdm
from skimage import color
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data

torch.autograd.set_detect_anomaly(True)


def bw_to_color(img_path: str,
                transformer: transforms.Compose,
                generator: Unet,
                device: torch.device) -> None:
    with torch.no_grad():
        generator.eval()
        img = Image.open(img_path).convert("RGB")
        img = transformer(img)
        arr = np.array(img)

        lab = color.rgb2lab(arr).astype("float32")
        ten_lab = transforms.ToTensor()(lab)

        L = ten_lab[0:1] / 50 - 1

        L = torch.unsqueeze(L, dim=0).to(device)

        fake_ab = generator(L)

        fake_ab = torch.squeeze(fake_ab, dim=0)
        L = torch.squeeze(L, dim=0)

        print(fake_ab.size())

        img_rgb = lab_to_rgb(L, fake_ab)

        im = Image.fromarray(img_rgb)
        im.show()


def compute_loss(preds: torch.Tensor,
                 loss_fn: nn.BCEWithLogitsLoss,
                 fake: bool,
                 device: torch.device):
    label_tensor = torch.zeros(size=preds.size(),
                               dtype=torch.float32,
                               device=device) if fake else torch.ones(size=preds.size(),
                                                                      dtype=torch.float32,
                                                                      device=device)

    return loss_fn(preds, label_tensor)


def initialize_weights(layer: torch.nn.Module,
                       gain=0.02):
    classname = layer.__class__.__name__
    if hasattr(layer, 'weight') and 'Conv' in classname:
        nn.init.normal_(layer.weight.data,
                        mean=0.0,
                        std=gain)

        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)

    elif 'BatchNorm2d' in classname:
        nn.init.normal_(layer.weight.data, 1., gain)
        nn.init.constant_(layer.bias.data, 0.)


def switch_gradients(model: nn.Module,
                     off_grads: bool):
    for param in model.parameters():
        param.requires_grad = off_grads


def optimize_discriminator(fake_ab: torch.Tensor,
                           real_ab: torch.Tensor,
                           L: torch.Tensor,
                           discriminator: Discriminator,
                           loss_fn: nn.BCEWithLogitsLoss,
                           device: torch.device):
    fake_image = torch.cat((L, fake_ab), dim=1)
    real_image = torch.cat((L, real_ab), dim=1)

    fake_preds = discriminator(fake_image.detach())
    real_preds = discriminator(real_image)

    fake_loss = compute_loss(fake_preds,
                             loss_fn,
                             True,
                             device)

    real_loss = compute_loss(real_preds,
                             loss_fn,
                             False,
                             device)

    full_loss = (fake_loss + real_loss) * 0.5

    return full_loss


def optimize_generator(fake_ab: torch.Tensor,
                       real_ab: torch.Tensor,
                       L: torch.Tensor,
                       discriminator: Discriminator,
                       loss_bce: nn.BCEWithLogitsLoss,
                       loss_l1: nn.L1Loss,
                       lambd: float,
                       device: torch.device):
    fake_image = torch.cat((L, fake_ab), dim=1)

    disc_pred = discriminator(fake_image)

    real_loss = compute_loss(disc_pred,
                             loss_bce,
                             False,
                             device)

    l1_loss = loss_l1(fake_ab, real_ab) * lambd
    full_loss = real_loss + l1_loss

    return full_loss


class TrainModel:

    def __init__(self,
                 train_loader: data.DataLoader,
                 test_loader: data.DataLoader,
                 val_loader: data.DataLoader,
                 lr_gen: float,
                 lr_d: float,
                 b_1: float,
                 b_2: float,
                 lambda_l1: float,
                 model_state_dict=None):

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__trl = train_loader
        self.__tsl = test_loader
        self.__val = val_loader

        self.__generator = Unet(256, 64, 0.5, False).to(device=self.__device)
        self.__discriminator = Discriminator(64, 3).to(device=self.__device)

        self.__loss_bce = nn.BCEWithLogitsLoss().to(self.__device)
        self.__loss_l1 = nn.L1Loss().to(self.__device)

        self.__discriminator.apply(initialize_weights)
        self.__generator.apply(initialize_weights)

        self.__optim_gen = torch.optim.Adam(self.__generator.parameters(),
                                            lr=lr_gen,
                                            betas=(b_1, b_2))

        self.__optim_disc = torch.optim.Adam(self.__discriminator.parameters(),
                                             lr=lr_d,
                                             betas=(b_1, b_2))

        self.__lambda = lambda_l1

        if model_state_dict is not None:
            checkpoint = torch.load(model_state_dict)

            self.__generator.load_state_dict(checkpoint["generator"])
            self.__discriminator.load_state_dict(checkpoint["discriminator"])
            self.__optim_disc.load_state_dict(checkpoint["optimizer_disc"])
            self.__optim_gen.load_state_dict(checkpoint["optimizer_gen"])

    def train_one_epoch(self,
                        epoch: int):
        self.__discriminator.train()
        self.__generator.train()

        gen_loss_list = torch.zeros(size=(1, len(self.__trl)),
                                    dtype=torch.float32,
                                    device=self.__device)

        disc_loss_list = torch.zeros(size=(1, len(self.__trl)),
                                     dtype=torch.float32,
                                     device=self.__device)

        count = 0
        for L, ab in tqdm(self.__trl, desc="Train", unit="batch"):
            ab = ab.to(self.__device)
            L = L.to(self.__device)

            fake_ab = self.__generator(L)

            self.__discriminator.train()
            switch_gradients(self.__discriminator, True)
            disc_loss = optimize_discriminator(fake_ab,
                                               ab,
                                               L,
                                               self.__discriminator,
                                               self.__loss_bce,
                                               self.__device)

            self.__optim_disc.zero_grad()
            disc_loss.backward()
            self.__optim_disc.step()

            self.__generator.train()
            switch_gradients(self.__discriminator, False)
            gen_loss = optimize_generator(fake_ab,
                                          ab,
                                          L,
                                          self.__discriminator,
                                          self.__loss_bce,
                                          self.__loss_l1,
                                          self.__lambda,
                                          self.__device)

            self.__optim_gen.zero_grad()
            gen_loss.backward()
            self.__optim_gen.step()

            gen_loss_list[:, count] += gen_loss
            disc_loss_list[:, count] += disc_loss

            count += 1

        form_str = "Train - Epoch {}: Gen loss {}. Disc Loss {}".format(epoch,
                                                                        torch.mean(gen_loss_list),
                                                                        torch.mean(disc_loss_list))

        print(form_str)

    def test_one_epoch(self,
                       epoch):

        gen_loss_list = torch.zeros(size=(1, len(self.__tsl)),
                                    dtype=torch.float32,
                                    device=self.__device)

        count = 0

        with torch.no_grad():
            self.__generator.eval()
            for L, ab in tqdm(self.__tsl, desc="Test", unit="batch"):
                ab = ab.to(self.__device)
                L = L.to(self.__device)

                fake_ab = self.__generator(L)

                l1_loss = self.__loss_l1(fake_ab, ab)

                gen_loss_list[:, count] += l1_loss

                count += 1

        form_str = "Test - Epoch {}: Gen loss {}.".format(epoch,
                                                          torch.mean(gen_loss_list))
        print(form_str)

    def validate_model(self):

        gen_loss_list = torch.zeros(size=(1, len(self.__val)),
                                    dtype=torch.float32,
                                    device=self.__device)

        count = 0

        with torch.no_grad():
            for L, ab in tqdm(self.__val, desc="Val", unit="batch"):
                ab = ab.to(self.__device)
                L = L.to(self.__device)

                fake_ab = self.__generator(L)

                l1_loss = self.__loss_l1(fake_ab, ab)

                gen_loss_list[:, count] += l1_loss

                count += 1

        form_str = "Val - Model: Gen loss {}.".format(torch.mean(gen_loss_list))
        print(form_str)

    def get_generator(self) -> Unet:
        return self.__generator

    def get_optim(self):
        return self.__optim_gen, self.__optim_disc

    def save_generator(self,
                       path: str,
                       epoch: int):

        torch.save({"generator": self.__generator.state_dict(),
                    "discriminator": self.__discriminator.state_dict(),
                    "optimizer_disc": self.__optim_disc.state_dict(),
                    "optimizer_gen": self.__optim_gen.state_dict(),
                    "epoch": epoch}, path)

    def get_device(self):
        return self.__device


if __name__ == '__main__':
    bw_file_path = "D:/Downloads/cast.jpg"

    size = 256
    batch_size = 16

    transformer_list = [transforms.Resize((size, size), transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(size)]

    val_ds = ColorizationDataset("../Data/Coco/2014/val2014",
                                 2_000,
                                 transforms.Compose(transformer_list))

    test_ds = ColorizationDataset("../Data/Coco/2014/test2014",
                                  2_000,
                                  transforms.Compose(transformer_list))

    transformer_list.append(transforms.RandomHorizontalFlip())

    train_ds = ColorizationDataset("../Data/Coco/2014/Train/train2014",
                                   8_000,
                                   transforms.Compose(transformer_list))

    train_loader = data.DataLoader(train_ds,
                                   batch_size=batch_size,
                                   shuffle=True)

    test_loader = data.DataLoader(train_ds,
                                  batch_size=batch_size,
                                  shuffle=True)

    val_loader = data.DataLoader(train_ds,
                                 batch_size=batch_size,
                                 shuffle=True)

    trainer = TrainModel(train_loader,
                         test_loader,
                         val_loader,
                         2e-4,
                         2e-4,
                         0.5,
                         0.999,
                         100.0,
                         "./SavedModels/Restorer.model")

    EPOCHS = 100
    gen = trainer.get_generator()
    bw_to_color(bw_file_path,
                transforms.Compose(transformer_list[:-2]),
                gen,
                device=trainer.get_device())

    # for epoch in range(EPOCHS):
    #     trainer.train_one_epoch(epoch)
    #     print()
    #
    #     if epoch != 0 and epoch % 10 == 0:
    #         trainer.test_one_epoch(epoch)
    #
    #     trainer.save_generator("./SavedModels/Restorer.model",
    #                            epoch)
    #
    #     gen = trainer.get_generator()
    #     bw_to_color(bw_file_path,
    #                 transforms.Compose(transformer_list[:-2]),
    #                 gen,
    #                 device=trainer.get_device())
    #
    # trainer.validate_model()
