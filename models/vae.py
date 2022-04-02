from models.basic_models.linear import Encoder, Decoder

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, VAEP, device=None):
        super(VAE, self).__init__()
        self.config = VAEP
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.checkpoint_file = self.config.checkpoint_file

        self.encoder = Encoder(self.config.encoder_params, device=self.device)
        self.decoder = Decoder(self.config.decoder_params, device=self.device)

        self.mse_criterion = nn.MSELoss(reduction="sum")
        self.prior = {
            "mean": torch.zeros(self.config.embed_dim, device=self.device),
            "log_sigma": torch.zeros(self.config.embed_dim, device=self.device)
        }

    def forward(self, x, use_epsilon=True):
        encoder_out = self.encoder(x)
        mean = encoder_out[:, :self.config.embed_dim]
        log_sigma = encoder_out[:, self.config.embed_dim:]
        z = mean
        if use_epsilon:
            epsilon = torch.randn(
                x.shape[0], self.config.embed_dim, device=self.device)
            z = mean + torch.exp(log_sigma) * epsilon
        decoder_out = self.decoder(z)
        return {
            "reconstruction": decoder_out,
            "mean": mean,
            "log_sigma": log_sigma
        }

    def loss(self, vae_outputs, target_images, beta):
        reconstruction_images = vae_outputs["reconstruction"]
        vae_mean = vae_outputs["mean"]
        vae_log_sigma = vae_outputs["log_sigma"]

        rec_loss = self.mse_criterion(
            target_images, reconstruction_images) / target_images.shape[0]
        kl_loss = torch.mean(torch.sum(
            VAE.kl_divergence(
                vae_mean, vae_log_sigma, self.prior["mean"], self.prior["log_sigma"]), dim=1), dtype=torch.float)

        return (rec_loss + beta * kl_loss).type(torch.float), rec_loss, kl_loss

    def reconstruct(self, x):
        encoder_out = self.encoder(x)
        mean = encoder_out[:, :self.config.embed_dim]
        log_sigma = encoder_out[:, self.config.embed_dim:]
        decoder_out = self.decoder(mean)
        return {
            "reconstruction": decoder_out,
            "mean": mean,
            "log_sigma": log_sigma
        }

    @staticmethod
    def kl_divergence(mu1, log_sigma1, mu2, log_sigma2):
        return (log_sigma2 - log_sigma1) + (torch.exp(log_sigma1) ** 2 + (mu1 - mu2) ** 2) \
            / (2 * torch.exp(log_sigma2) ** 2) - 0.5

