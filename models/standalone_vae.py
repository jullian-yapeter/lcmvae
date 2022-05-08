import torch
import torch.nn as nn
from models.basic_models.conv import ConvDecoder768
from models.basic_models.linear import Encoder, Decoder

import masks 


#####################################################################
# Encoder Architecture: in_dims = (3, 224, 224)                     #
#   - Conv2d, kernel: 16, stride: 4, pad=0, out_dims: (80, 56, 56)  #
#   - LeakyReLU                                                     #
#   - Conv2d, kernel: 8, stride: 3, pad=0, out_dims: (160, 16, 16)   #
#   - BatchNorm2d                                                   #
#   - LeakyReLU                                                     #
#   - Conv2d, kernel: 4, stride: 2, pad=0, out_dims: (320, 7, 7)    #
#   - BatchNorm2d                                                   #
#   - LeakyReLU                                                     #
#   - Conv2d, kernel: 3, stride: 2, pad=0, out_dims: (640, 3, 3)    #
#   - BatchNorm2d                                                   #
#   - LeakyReLU                                                     #
#   - Conv2d, kernel: 3, stride: 1, pad=0, out_dims: (1280, 1, 1)    #
#   - BatchNorm2d                                                   #
#   - LeakyReLU                                                     #
#   - Flatten                                                       #
#   - Linear: out_dim: embed_dim                                    #
#####################################################################
class Conv2d_Alone(nn.Module):
    def __init__(self, out_dim):
        super(Conv2d_Alone, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=80, kernel_size=16,
                      stride=4, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=80, out_channels=160, kernel_size=8,
                      stride=3, padding=0, bias=True),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=160, out_channels=320, kernel_size=4,
                      stride=2, padding=0, bias=True),
            nn.BatchNorm2d(320),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=320, out_channels=640, kernel_size=3,
                      stride=2, padding=0, bias=True),
            nn.BatchNorm2d(640),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=640, out_channels=1280, kernel_size=3,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1280, out_dim, bias=True)
        )

    def forward(self, x):
        return self.net(x)


class Encoder_Alone(nn.Module):
    def __init__(self, embed_dim, device=None):
        super(Encoder_Alone, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.embed_dim = embed_dim
        self.conv2d = Conv2d_Alone(embed_dim * 2).to(self.device)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class StandAloneVAE(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.checkpoint_file = config.checkpoint_file

        if self.config.mask_type == 'Patch':
            self.Mask = masks.PatchMask(self.config.mask_ratio, 16)
        elif self.config.mask_type == 'Pixel':
            self.Mask = masks.PixelMask(self.config.mask_ratio)

        self.encoder = Encoder_Alone(self.config.embed_dim, device=self.device)

        if self.config.use_linear_decoder:
            self.decoder = Decoder(self.config.decoder_params, device=self.device)
        else:
            self.decoder = ConvDecoder768(
                self.config.embed_dim, out_channels=3, device=self.device)

        self.mse_criterion = nn.MSELoss(reduction="sum")
        self.prior = {
            "mean": torch.zeros(config.embed_dim, device=self.device),
            "log_sigma": torch.zeros(config.embed_dim, device=self.device)
        }

    def forward(self, x):
        if self.config.mask_type != 'None':
            x, mask = self.Mask(x)[0]

        encoder_out = self.encoder(x)
        mean = encoder_out[:, :self.config.embed_dim]
        log_sigma = encoder_out[:, self.config.embed_dim:]
        z = mean
        if self.config.use_epsilon:
            epsilon = torch.randn(
                x.shape[0], self.config.embed_dim, device=self.device)
            z = mean + torch.exp(log_sigma) * epsilon
        decoder_out = self.decoder(z)
        if self.config.mask_type == 'None':
            return {
                "reconstruction": decoder_out,
                "mean": mean,
                "log_sigma": log_sigma,
                "z": z
            }
        else:
            return {
                "reconstruction": decoder_out,
                "mean": mean,
                "log_sigma": log_sigma,
                "z": z,
                'mask': mask,
                'masked_img': x
            }

    def loss(self, vae_outputs, target_images, beta):
        print(type(vae_outputs))
        reconstruction_images = vae_outputs["reconstruction"]
        vae_mean = vae_outputs["mean"]
        vae_log_sigma = vae_outputs["log_sigma"]

        rec_loss = self.mse_criterion(
            target_images, reconstruction_images) / target_images.shape[0]
        kl_loss = torch.mean(torch.sum(
            StandAloneVAE.kl_divergence(
                vae_mean, vae_log_sigma, self.prior["mean"], self.prior["log_sigma"]), dim=1), dtype=torch.float32)
        if self.config.use_epsilon:
            return (rec_loss + beta * kl_loss).type(torch.float32), rec_loss, kl_loss
        return rec_loss, rec_loss, kl_loss

    def reconstruct(self, x):
        if self.config.mask_type is not None:
            x, mask = self.Mask(x)

        encoder_out = self.encoder(x)
        mean = encoder_out[:, :self.config.embed_dim]
        log_sigma = encoder_out[:, self.config.embed_dim:]
        decoder_out = self.decoder(mean)
        if self.config.mask_type == 'None':
            return {
                "reconstruction": decoder_out,
                "mean": mean,
                "log_sigma": log_sigma
            }
        else:
            return {
                "reconstruction": decoder_out,
                "mean": mean,
                "log_sigma": log_sigma,
                "mask": mask,
                'masked_img': x
            }

    @staticmethod
    def kl_divergence(mu1, log_sigma1, mu2, log_sigma2):
        return (log_sigma2 - log_sigma1) + (torch.exp(log_sigma1) ** 2 + (mu1 - mu2) ** 2) \
               / (2 * torch.exp(log_sigma2) ** 2) - 0.5

