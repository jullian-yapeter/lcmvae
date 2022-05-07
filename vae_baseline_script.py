import torch
import torch.nn as nn
import pandas as pd
from models.basic_models.conv import ConvDecoder768
from models.basic_models.linear import Encoder, Decoder
from models.params import STANDALONE_VAE_PARAMS as SVAEP
from params import PRETRAIN_DATASET_PARAMS
from dataset import MyCocoCaption, MyCocoCaptionDetection

import masks
from models.basic_models.params import LINEAR_NETWORK_PARAMS, DECODER_PARAMS

import math
import numpy as np
import sys


class STANDALONE_VAE_PARAMS:
    checkpoint_file = ''
    im_dims = [3, 224, 224]
    embed_dim = 768
    use_linear_decoder = False
    # use_prev_conv_layer = True
    use_epsilon = not ('noVar' in sys.argv)
    mask_type = sys.argv[2]  # 'Patch' 'Pixel'
    mask_ratio = 0.75

    decoder_params = DECODER_PARAMS()
    decoder_params.im_dims = (3, 224, 224)
    decoder_params.linear_params.output_dim = embed_dim
    decoder_params.linear_params.activation = nn.LeakyReLU()
    decoder_params.linear_params.linear_layer_params = [
        {"in_dim": embed_dim, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": np.prod(im_dims)}
    ]


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


class StandaloneVAE(nn.Module):
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
            x, mask = self.Mask(x)

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
        reconstruction_images = vae_outputs["reconstruction"]
        vae_mean = vae_outputs["mean"]
        vae_log_sigma = vae_outputs["log_sigma"]

        rec_loss = self.mse_criterion(
            target_images, reconstruction_images) / target_images.shape[0]
        kl_loss = torch.mean(torch.sum(
            StandaloneVAE.kl_divergence(
                vae_mean, vae_log_sigma, self.prior["mean"], self.prior["log_sigma"]), dim=1), dtype=torch.float32)
        if self.config.use_epsilon:
            return (rec_loss + beta * kl_loss).type(torch.float32), rec_loss, kl_loss
        return rec_loss, rec_loss, kl_loss

    def reconstruct(self, x):
        if self.config.mask_type != 'None':
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

from utils import save_checkpoint, save_model




import sys, os, inspect, time

import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import VAEPreTrainer
from dataset import MyCocoCaption, MyCocoCaptionDetection

from models.basic_models.conv import ConvDecoder768

from params import PRETRAIN_PARAMS as PTP
from params import PRETRAIN_DATASET_PARAMS

from utils import denormalize_torch_to_cv2, count_parameters
import os

experiment_name = sys.argv[1] + time.strftime("_%m%d_%H%M")
print('-' * 40);
print("Experiment: ", experiment_name);
print('-' * 40)


device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

# # Construct Dataset
# coco_val2017 = MyCocoCaption(root = PRETRAIN_DATASET_PARAMS.image_dir,
#                             annFile = PRETRAIN_DATASET_PARAMS.ann_file,
#                             from_pretrained = PRETRAIN_DATASET_PARAMS.from_pretrained)

# detection dataset: outputs: img, (caption, mask)
# cats = {1: 'person', 2: 'bicycle', 3: 'car',4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat'}
coco_val2017 = MyCocoCaptionDetection(root=PRETRAIN_DATASET_PARAMS.image_dir,
                                      annFile=PRETRAIN_DATASET_PARAMS.ann_file,
                                      detAnnFile=PRETRAIN_DATASET_PARAMS.det_ann_file,
                                      superclasses=["person", "vehicle"],
                                      from_pretrained=PRETRAIN_DATASET_PARAMS.from_pretrained)

# image mean and std for reconstruction
image_mean = torch.tensor(coco_val2017.feature_extractor.image_mean)
image_std = torch.tensor(coco_val2017.feature_extractor.image_std)

# Check the info of dataset, you can ignore this part
print('-' * 40)
coco_val2017.coco.info()
print('-' * 40)
print(f'The number of samples: {len(coco_val2017)}')
first_img, (first_cap, first_segment) = coco_val2017[0]
print(f'Image shape: {first_img.size()}')

save_dir = f"./saved_models/{experiment_name}"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
with open(f"{save_dir}/params_{experiment_name}.py", 'w+') as f:
    f.write(f"# PARAMS for Experiment: {experiment_name}\n")
    f.write(f"# GPU Type: {torch.cuda.get_device_name()}\n\n")
    f.write(
        "from models.basic_models.params import LINEAR_NETWORK_PARAMS, DECODER_PARAMS\n"
        "from utils import has_internet\n"
        "import math, torch, torch.nn as nn\n\n")
    lines = map(inspect.getsource, [
        PTP, PRETRAIN_DATASET_PARAMS, STANDALONE_VAE_PARAMS])
    f.write('\n\n'.join(lines))

# Build Dataloader for pretrain
data_loader = DataLoader(dataset=coco_val2017,
                         batch_size=PRETRAIN_DATASET_PARAMS.batch_size,
                         shuffle=PRETRAIN_DATASET_PARAMS.shuffle,
                         num_workers=PRETRAIN_DATASET_PARAMS.num_workers)

# # Check: print info for each batch
# i = 0
# for imgs, (caps, segment) in data_loader:
#     print(f'batch_{i}')
#     print(f"Image batch shape: {imgs.size()}")
#     print(f"Segmentation batch shape: {segment.size()}")
#     print(f"Caption batch shape: {len(caps)}")
#     i += 1
# exit()

# lcmvae = LCMVAE(LCMVAEP, device=device)
svae = StandaloneVAE(STANDALONE_VAE_PARAMS, device=device)

count_parameters(svae)

pretrainer = VAEPreTrainer(svae, PTP, experiment_name=experiment_name + "_pretrain", save_dir=save_dir)
pretrainer.run(data=data_loader)


