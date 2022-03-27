from models.basic_models.linear import Decoder

import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    def __init__(self, decoder_params, im_dims=(224, 224, 3), device=None):
        super(ReconstructionHead, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.checkpoint_file = "reconstruction_head"
        self.model = Decoder(decoder_params, device=device)
        self.im_dims = im_dims

    def forward(self, lcmvae_outputs):
        out = self.model(lcmvae_outputs["mean"])
        return out.view(-1, *self.im_dims)


class SegmentationHead(nn.Module):
    def __init__(self, decoder_params, im_dims=(224, 224, 3), device=None):
        super(SegmentationHead, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.checkpoint_file = "reconstruction_head"
        self.model = Decoder(decoder_params, device=device)
        self.im_dims = im_dims

    def forward(self, x):
        out = self.model(x)
        return out.view(-1, *self.im_dims)
