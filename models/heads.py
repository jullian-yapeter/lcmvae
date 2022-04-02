from models.basic_models.linear import Decoder

import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    # NOTE: im_dims should be (3,224,224)
    def __init__(self, decoder_params, im_dims=(3, 224, 224), device=None):
        super(ReconstructionHead, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.checkpoint_file = "reconstruction_head"
        self.model = Decoder(decoder_params, device=device)
        self.im_dims = im_dims

    def forward(self, lcmvae_outputs):
        out = self.model(lcmvae_outputs["mean"])
        return out.view(-1, *self.im_dims)


class ConvDecoder512(nn.Module):
  """
  Convoluted Decoder for LCMVAE. (Temporary)
  in_dim:  
  """

  def __init__(self, config, device=None):
    super().__init__()
    self.config = config
    self.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.checkpoint_file = self.config.checkpoint_file
    self.embed_dim = config.embed_dim
    self.out_channels = config.out_channels

    self.map = nn.Linear(self.config.embed_dim, 512, bias=True).to(
        self.device)   # for initial Linear layer
    self.net = nn.Sequential(
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3,
                           stride=1, padding=0, bias=True),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                           stride=2, padding=0, bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                           stride=2, padding=0, bias=True),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=8,
                           stride=3, padding=0, bias=True),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(
            in_channels=32, out_channels=self.out_channels, kernel_size=16,
            stride=4, padding=0, bias=True)
    ).to(self.device)

  def forward(self, x):
    return self.net(self.map(x).reshape(-1, 512, 1, 1))
