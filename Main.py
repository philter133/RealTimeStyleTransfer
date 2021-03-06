import torch
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utilities import display_image, load_image, compute_gram, get_transformer_list, transform_image
from Models.VGG import VGG16
from Models.Transformer import TransformerNetwork

# Set Random Seed
torch.manual_seed(108)

# Adds style in real time by training this model
class RealTimeStyleTransfer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 style_weight: float,
                 content_weight: float,
                 tvr_weight: float,
                 style_image_path: str,
                 train_data_path: str,
                 content_loss_fn: torch.nn.MSELoss,
                 style_loss_fn: torch.nn.MSELoss,
                 normalize: bool,
                 range_255: bool):

        self.__normalize = normalize
        self.__range_255 = range_255

        # Set training device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__device = torch.device(device=device)

        # Hyper_Parameters
        self.__batch_size = batch_size
        self.__lr = learning_rate
        self.__lambda_s = style_weight
        self.__lambda_c = content_weight
        self.__lambda_tvr = tvr_weight

        # Models
        self.__vgg = VGG16().to(self.__device)
        self.__transformer = TransformerNetwork().to(self.__device)

        # Image_Transformers
        image_tf = transforms.Compose(get_transformer_list(normalize, range_255))

        style_tf = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

        self.__normalizer_tf = transforms.Compose([transforms.Lambda(lambda x: x / 255),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])

        train_ds = datasets.ImageFolder(train_data_path,
                                        image_tf)

        self.__data_loader = DataLoader(train_ds,
                                        batch_size=self.__batch_size,
                                        shuffle=True)

        style_image = load_image(style_image_path)
        style_tensor = style_tf(style_image).to(self.__device)

        style_tensor = style_tensor.repeat(self.__batch_size, 1, 1, 1)

        layer_dict = self.__vgg(style_tensor)

        # Compute gram matrices
        self.__style_gram = {}

        for key in list(layer_dict.keys()):
            self.__style_gram[key] = compute_gram(layer_dict[key])

        self.__loss_fn_content = content_loss_fn.to(self.__device)
        self.__loss_fn_style = style_loss_fn.to(self.__device)

        self.__optim = torch.optim.Adam(self.__transformer.parameters(),
                                        self.__lr)

    def save_model(self, save_path: str):
        torch.save({
            'model_state_dict': self.__transformer.state_dict(),
            'optimizer_state_dict': self.__optim.state_dict(),
        }, save_path)

    def transform_image(self, content_image_path):
        self.__transformer.eval()

        tf = transforms.Compose(get_transformer_list(self.__normalize, self.__range_255)[2:])

        content_image = load_image(content_image_path)
        content_tensor = torch.unsqueeze(tf(content_image), dim=0).to(self.__device)

        display_image(torch.squeeze(self.__transformer(content_tensor), dim=0).cpu(),
                      range_255=self.__range_255,
                      normalize=self.__normalize)

    def train_model_one_epoch(self):

        # Lists to check if the model converges
        style_loss_list = []
        feature_loss_list = []
        tv_loss_list = []
        total_loss_list = []

        self.__transformer.train()

        count = 0

        for image_batch, _ in tqdm(self.__data_loader):
            torch.cuda.empty_cache()
            self.__optim.zero_grad()
            count += 1

            batch_size = image_batch.size()[0]

            y = image_batch.to(self.__device)

            # Calculate Content Loss

            y_hat = self.__transformer(y)

            if self.__range_255 and not self.__normalize:
                fi_y_hat = self.__vgg(self.__normalizer_tf(y_hat))
                fi_y = self.__vgg(self.__normalizer_tf(y))
            else:
                fi_y_hat = self.__vgg(y_hat)
                fi_y = self.__vgg(y)

            content_loss = self.__lambda_c * self.__loss_fn_content(fi_y_hat["relu2_2"],
                                                                    fi_y["relu2_2"])
            feature_loss_list.append(content_loss)

            # Calculate Style Loss

            style_loss = 0.0

            for key in list(fi_y_hat.keys()):
                style_loss += self.__loss_fn_style(compute_gram(fi_y_hat[key]),
                                                   self.__style_gram[key][:batch_size])

            style_loss *= self.__lambda_s

            style_loss_list.append(style_loss)

            # Total Variation Regularization

            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = self.__lambda_tvr * (diff_i + diff_j)
            tv_loss_list.append(tv_loss)

            total_loss = tv_loss + style_loss + content_loss

            total_loss_list.append(total_loss)

            total_loss.backward()

            self.__optim.step()

            if count % 1000 == 0:
                self.transform_image("./Data/ContentTestImages/karya.jpg")
                self.__transformer.train()


if __name__ == '__main__':
    # style_transfer = RealTimeStyleTransfer(4,
    #                                        1e-3,
    #                                        1e10,
    #                                        1e5,
    #                                        0,
    #                                        "./Data/Style/mosaic.jpg",
    #                                        "./Data/Coco/2014/Train",
    #                                        torch.nn.MSELoss(),
    #                                        torch.nn.MSELoss(),
    #                                        False,
    #                                        True)
    #
    # epochs = 2
    #
    # for epoc in range(epochs):
    #     style_transfer.train_model_one_epoch()
    #
    # style_transfer.transform_image("./Data/ContentTestImages/test1.jpg")
    # style_transfer.save_model("./SavedModels/mosaic.model")

    style_transfer = transform_image("./SavedModels/starrynight.model",
                                     "./Data/ContentTestImages/castle.jpg",
                                     save_path="D:/RealTimeStyleTransfer/SavedPics/scene.jpg",
                                     save=True,
                                     normalize=False,
                                     range_255=True)
