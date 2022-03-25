from models.basic_models.params import LINEAR_NETWORK_PARAMS

import math
import torch.nn as nn

class VAE_PARAMS:
    checkpoint_file = "vae"
    embed_dim = 768
    im_dims = [224, 224, 3]

    encoder_params = LINEAR_NETWORK_PARAMS()
    encoder_params.output_dim = embed_dim * 2
    encoder_params.activation = nn.LeakyReLU()
    encoder_params.linear_layer_params = [
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 768},
        {"in_dim": 768, "out_dim": 768},
        {"in_dim": 768, "out_dim": 768},
        {"in_dim": 768, "out_dim": encoder_params.output_dim}
    ]

    decoder_params = LINEAR_NETWORK_PARAMS()
    decoder_params.output_dim = embed_dim
    decoder_params.activation = nn.LeakyReLU()
    decoder_params.linear_layer_params = [
        {"in_dim": embed_dim, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": math.prod(im_dims)}
    ]


class LCMVAE_PARAMS:
    checkpoint_file = "lcmvae"
    is_mae = True
    mask_ratio = 0.75
    vae_params = VAE_PARAMS()
