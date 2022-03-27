from models.basic_models.params import LINEAR_NETWORK_PARAMS as LNP

import torch
import torch.nn as nn


class LinearNetwork(nn.Module):
    def __init__(self, LNP, device=None):
        super(LinearNetwork, self).__init__()
        self.config = LNP
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        self.activation = self.config.activation
        linear_layers = nn.ModuleList()
        for i, llp in enumerate(self.config.linear_layer_params):
            linear_layers.append(nn.Linear(llp["in_dim"], llp["out_dim"]))
            if i < len(self.config.linear_layer_params) - 1:
                linear_layers.append(self.activation)

        self.model = nn.Sequential(
            *linear_layers
        ).to(self.device)

    def forward(self, x):
        x.to(self.device)
        out = self.model(x)
        return out

class Encoder(LinearNetwork):
    def __init__(self, encoder_params, device=None):
        super(Encoder, self).__init__(encoder_params, device=device)
        self.checkpoint_file = "encoder"


class Decoder(nn.Module):
    def __init__(self, decoder_params, device=None):
        super(Decoder, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.checkpoint_file = "decoder"
        self.model = nn.Sequential(
            LinearNetwork(decoder_params),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, x):
        return self.model(x)
