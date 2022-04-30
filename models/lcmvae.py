from models.frozen_transformers import ImageCaptionEncoder
from models.vae import VAE
from models.heads import LatentReconstructor

import torch
import torch.nn as nn


class LCMVAE(nn.Module):
    def __init__(self, LCMVAEP, device=None):
        super(LCMVAE, self).__init__()
        self.config = LCMVAEP
        self.checkpoint_file = self.config.checkpoint_file
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.im_cap_encoder = ImageCaptionEncoder(
            is_mae=self.config.is_mae,
            mask_ratio=self.config.mask_ratio,
            mode=self.config.mae_mode,
            no_caption=self.config.no_caption,
            device=device)
        self.vae = VAE(self.config.vae_params, device=device)
        self.vae.apply(LCMVAE._init_vae_weights)
        if self.config.use_latent_regularizer:
            self.latent_reconstructor = LatentReconstructor(
                self.config.latent_reconstructor_params, device=device)
            self.latent_reconstructor_loss = nn.MSELoss()

    def forward(self, images, captions):
        mask = None
        with torch.no_grad():
            if self.config.is_mae:
                im_cap_embedding, mask = self.im_cap_encoder.forward(images, captions)
            else:
                im_cap_embedding = self.im_cap_encoder.forward(images, captions)
        language_embedding = im_cap_embedding[:, -self.config.embed_dim:]
        vae_outputs = self.vae(im_cap_embedding)
        if self.config.use_latent_regularizer:
            latent_reconstruction = self.latent_reconstructor(vae_outputs["z"])
            vae_outputs.update(
                {"latent_reconstruction": latent_reconstruction, "latent_target": language_embedding})
        return vae_outputs, mask

    def loss(self, vae_outputs, target_images, beta, delta=None):
        vae_loss = self.vae.loss(vae_outputs, target_images, beta)
        if self.config.use_latent_regularizer:
            latent_reconstruction_loss = self.latent_reconstructor_loss(
                vae_outputs["latent_reconstruction"], vae_outputs["latent_target"])
            return vae_loss[0] + delta * latent_reconstruction_loss, *vae_loss[1:], latent_reconstruction_loss
        return vae_loss

    def reconstruct(self, images, captions):
        mask = None
        with torch.no_grad():
            if self.config.is_mae:
                im_cap_embedding, mask = self.im_cap_encoder.forward(images, captions)
            else:
                im_cap_embedding = self.im_cap_encoder.forward(images, captions)
        vae_outputs = self.vae.reconstruct(im_cap_embedding)
        return vae_outputs, mask

    def run(self, images, captions):
        outputs, mask = self.reconstruct(images, captions)
        return outputs["reconstruction"][0], mask

    @staticmethod
    def _init_vae_weights(m):
        try:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        except:
            pass


# class LCMVAEDownstream(nn.Module):
#     def __init__(self, LCMVAEDP, head, criterion, device=None):
#         super(LCMVAEDownstream, self).__init__()
#         self.config = LCMVAEDP
#         self.checkpoint_file = self.config.checkpoint_file
#         self.device = torch.device(
#             'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
#         self.im_cap_encoder = ImageCaptionEncoder(
#             is_mae=self.config.is_mae, mask_ratio=self.config.mask_ratio, device=device)
#         self.encoder = Encoder(self.config.encoder_params, device=device)
#         self.head = head
#         self.criterion = criterion

#     def forward(self, images, captions):
#         mask = None
#         with torch.no_grad():
#             if self.config.is_mae:
#                 im_cap_embedding, mask = self.im_cap_encoder.forward(
#                     images, captions)
#             else:
#                 im_cap_embedding = self.im_cap_encoder.forward(
#                     images, captions)
#         enc_outputs = self.encoder(im_cap_embedding)
#         head_outputs = self.head(enc_outputs[:, :self.config.embed_dim])
#         return head_outputs, mask

#     def loss(self, targets, head_outputs):
#         return self.criterion(targets, head_outputs)