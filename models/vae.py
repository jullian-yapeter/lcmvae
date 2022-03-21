from models.basic_models.linear import LinearNetwork

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, VAEP, device=None):
        super(VAE, self).__init__()
        self.config = VAEP
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.encoder = LinearNetwork(self.config.encoder_params)
        self.decoder = nn.Sequential(
            LinearNetwork(self.config.decoder_params),
            nn.Sigmoid()
        )

    def forward(self, x):
        x.to(self.device)
        encoder_out = self.encoder(x)
        mean = encoder_out[:, :self.config.embed_dim]
        log_sigma = encoder_out[:, self.config.embed_dim:]
        epsilon = torch.randn(
            x.shape[0], self.config.embed_dim).to(self.device)
        z = mean + torch.exp(log_sigma) * epsilon
        decoder_out = self.decoder(z)
        return decoder_out.view(-1, *self.config.im_dims)
