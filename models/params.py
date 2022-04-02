from json import decoder
from models.basic_models.params import LINEAR_NETWORK_PARAMS, DECODER_PARAMS

import math
import torch.nn as nn

class VAE_PARAMS:
    checkpoint_file = "vae"
    embed_dim = 768
    im_dims = (3, 224, 224)

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

    decoder_params = DECODER_PARAMS()
    decoder_params.im_dims = (3, 224, 224)
    decoder_params.linear_params.output_dim = embed_dim
    decoder_params.linear_params.activation = nn.LeakyReLU()
    decoder_params.linear_params.linear_layer_params = [
        {"in_dim": embed_dim, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": math.prod(im_dims)}
    ]


class CONV_VAE_PARAMS:
    checkpoint_file = "conv_vae"
    embed_dim = 512
    im_dims = (3, 224, 224)

    encoder_params = LINEAR_NETWORK_PARAMS()
    encoder_params.output_dim = embed_dim * 2
    encoder_params.activation = nn.LeakyReLU()
    encoder_params.linear_layer_params = [
        {"in_dim": 1536, "out_dim": 1024},
        {"in_dim": 1024, "out_dim": 512},
        {"in_dim": 512, "out_dim": 512},
        {"in_dim": 512, "out_dim": 512},
        {"in_dim": 512, "out_dim": encoder_params.output_dim}
    ]
    ## TODO: Add decoder params for hidden units/channells


class LCMVAE_PARAMS:
    is_mae = True
    mask_ratio = 0.75
    vae_params = VAE_PARAMS()
    no_caption = False
    checkpoint_file = "lcmvae_capless"  if no_caption else "lcmvae"  
    checkpoint_file = checkpoint_file


class CAPTIONLESS_LCMVAE_PARAMS:
    checkpoint_file = "captionless_lcmvae"
    embed_dim = 768
    im_dims = [3, 224, 224]


class STANDALONE_VAE_PARAMS:
    checkpoint_file = "standalone_vae"
    embed_dim = 768
    im_dims = [3, 224, 224]


class CONV_DECODER_512_PARAMS:
    checkpoint_file = "conv_decoder_512"
    embed_dim = 768
    out_channels = 10


