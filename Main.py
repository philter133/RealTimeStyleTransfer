import torch
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from Utils.utilities import view_image, load_image, compute_gram, save_image
from Models.VGG import VGG16
from Models.Transformer import TransformerNetwork

# Set Random Seed
torch.manual_seed(108)


class StyleTransfer:

    def __init__(self,
                 state_dict_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__device = torch.device(device=device)

        checkpoint = torch.load(state_dict_path)
        self.__model = TransformerNetwork().to(self.__device)
        self.__model.load_state_dict(checkpoint["model_state_dict"])

    def transform_image(self, content_image_path,save_path):
        with torch.no_grad():
            self.__model.eval()

            style_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

            content_image = load_image(content_image_path)
            content_tensor = torch.unsqueeze(style_tf(content_image), dim=0).to(self.__device)

            view_image(torch.squeeze(self.__model(content_tensor), dim=0).cpu())
            save_image(torch.squeeze(self.__model(content_tensor), dim=0).cpu(), save_path)




class RealTimeStyleTransfer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 style_weight: float,
                 content_weight: float,
                 tvr_weight: float,
                 style_image_path: str,
                 train_data_path: str):
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
        image_tf = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(256),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

        style_tf = transforms.Compose([transforms.ToTensor(),
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

        self.__loss_fn = torch.nn.MSELoss().to(self.__device)

        self.__optim = torch.optim.Adam(self.__transformer.parameters(),
                                        self.__lr)

    def save_model(self, save_path: str):
        torch.save({
            'model_state_dict': self.__transformer.state_dict(),
            'optimizer_state_dict': self.__optim.state_dict(),
        }, save_path)

    def transform_image(self, content_image_path):
        self.__transformer.eval()

        style_tf = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

        content_image = load_image(content_image_path)
        content_tensor = torch.unsqueeze(style_tf(content_image), dim=0).to(self.__device)

        view_image(torch.squeeze(self.__transformer(content_tensor), dim=0).cpu())

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
            fi_y_hat = self.__vgg(y_hat)
            fi_y = self.__vgg(y)

            content_loss = self.__lambda_c * self.__loss_fn(fi_y_hat["relu2_2"],
                                                            fi_y["relu2_2"])
            feature_loss_list.append(content_loss)

            # Calculate Style Loss

            style_loss = 0.0

            for key in list(fi_y_hat.keys()):
                style_loss += self.__loss_fn(compute_gram(fi_y_hat[key]),
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
                self.transform_image("./Data/ContentTestImages/city.jpg")
                self.__transformer.train()


if __name__ == '__main__':
    style_transfer = RealTimeStyleTransfer(4,
                                           1e-3,
                                           4e10,
                                           1e5,
                                           0,
                                           "./Data/Style/edtaonisl.png",
                                           "./Data/Coco/2014/Train")

    epochs = 2

    for epoc in range(epochs):
        style_transfer.train_model_one_epoch()

    style_transfer.transform_image("./Data/ContentTestImages/test1.jpg")
    style_transfer.save_model("./SavedModels/JuneTree.model")

    style_transfer = StyleTransfer("./SavedModels/JuneTree.model")
    style_transfer.transform_image("./Data/ContentTestImages/test2.jpg", "D:/RealTimeStyleTransfer/SavedPics/temp.jpg")
