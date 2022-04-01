from models.frozen_transformers import VitEncoder, VitMaeEncoder
from models.vae import VAE

import torch
import torch.nn as nn

# TODO: Need to be tested
class CaptionlessLCMVAE(nn.Module):
    def __init__(self, LCMVAEP, device=None):
        super().__init__()
        self.config = LCMVAEP
        self.checkpoint_file = self.config.checkpoint_file
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        if self.config.is_mae:
            self.img_encoder = VitMaeEncoder(mask_ratio=self.config.mask_ratio, device=self.device)
        else:
            self.img_encoder = VitEncoder(device=self.device)
        self.vae = VAE(self.config.vae_params, device=self.device)
        
    def forward(self, images, captions=None, pretraining=True):
        mask = None
        with torch.no_grad():
            if self.config.is_mae:
                img_embedding, mask = self.img_encoder.forward(images)
            else:
                img_embedding = self.img_encoder.forward(images)
        vae_outputs = self.vae(
            torch.cat((img_embedding, torch.zeros_like(img_embedding, device=self.device)), dim=-1),
            pretraining = pretraining)
        return vae_outputs, mask

    def loss(self, target_images, vae_outputs, beta):
        return self.vae.loss(target_images, vae_outputs, beta)

    def reconstruct(self, images, captions=None):
        mask = None
        with torch.no_grad():
            if self.config.is_mae:
                img_embedding, mask = self.img_encoder.forward(images)
            else:
                img_embedding = self.img_encoder.forward(images)
        vae_outputs = self.vae.reconstruct(img_embedding)
        return vae_outputs, mask

    def run(self, images, captions=None):
        outputs, mask = self.reconstruct(images)
        return outputs["reconstruction"][0].cpu().detach().numpy() * 255, mask

