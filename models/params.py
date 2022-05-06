from json import decoder
from models.basic_models.params import LINEAR_NETWORK_PARAMS, DECODER_PARAMS

import math
import torch.nn as nn
import numpy as np
class VAE_PARAMS:
    checkpoint_file = "vae"
    use_linear_decoder = True
    use_pre_conv_layer = False
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
        {"in_dim": 1536, "out_dim": np.prod(im_dims)}
    ]


class LATENT_RECONSTRUCTOR_PARAMS:
    checkpoint_file = "latent_reconstructor"
    embed_dim = 768

    decoder_params = LINEAR_NETWORK_PARAMS()
    decoder_params.output_dim = embed_dim
    decoder_params.activation = nn.LeakyReLU()
    decoder_params.linear_layer_params = [
        {"in_dim": embed_dim, "out_dim": 768},
        {"in_dim": 768, "out_dim": decoder_params.output_dim},
    ]

class CONV_VAE_PARAMS:
    checkpoint_file = "conv_vae"
    use_linear_decoder = False
    use_epsilon = False
    use_pre_conv_layer = False
    embed_dim = 768
    im_dims = (3, 224, 224)

    encoder_params = LINEAR_NETWORK_PARAMS()
    encoder_params.output_dim = embed_dim * 2
    encoder_params.activation = nn.LeakyReLU()
    encoder_params.linear_layer_params = [
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 768},
        {"in_dim": 768, "out_dim": encoder_params.output_dim}
    ]


class CONV_VAE_BIG_PARAMS:
    checkpoint_file = "conv_vae_big"
    use_linear_decoder = False
    use_epsilon = False
    use_pre_conv_layer = False
    embed_dim = 768
    im_dims = (3, 224, 224)

    encoder_params = LINEAR_NETWORK_PARAMS()
    encoder_params.output_dim = embed_dim * 2
    encoder_params.activation = nn.LeakyReLU()
    encoder_params.linear_layer_params = [
        {"in_dim": (196+1)*768, "out_dim": 3072},
        {"in_dim": 3072, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": encoder_params.output_dim}
    ]


# class SMALL_VAE_PARAMS:
#     checkpoint_file = "small_vae"
#     embed_dim = 256
#     im_dims = (3, 224, 224)

#     encoder_params = LINEAR_NETWORK_PARAMS()
#     encoder_params.output_dim = embed_dim * 2
#     encoder_params.activation = nn.LeakyReLU()
#     encoder_params.linear_layer_params = [
#         {"in_dim": 1536, "out_dim": 768},
#         {"in_dim": 768, "out_dim": 512},
#         {"in_dim": 512, "out_dim": 256},
#         {"in_dim": 256, "out_dim": 256},
#         {"in_dim": 256, "out_dim": encoder_params.output_dim}
#     ]

    # decoder_params = DECODER_PARAMS()
    # decoder_params.im_dims = (3, 224, 224)
    # decoder_params.linear_params.output_dim = embed_dim
    # decoder_params.linear_params.activation = nn.LeakyReLU()
    # decoder_params.linear_params.linear_layer_params = [
    #     {"in_dim": embed_dim, "out_dim": 256},
    #     {"in_dim": 256, "out_dim": 256},
    #     {"in_dim": 256, "out_dim": 256},
    #     {"in_dim": 256, "out_dim": 512},
    #     {"in_dim": 512, "out_dim": np.prod(im_dims)}
    # ]


class CD512P:
    checkpoint_file = "conv_decoder_512"
    embed_dim = 256
    out_channels = 10

class LCMVAE_PARAMS:
    checkpoint_file = "lcmvae"
    embed_dim = 768
    use_latent_regularizer = True
    use_epsilon = True
    use_pre_conv_layer = True
    is_mae = True
    use_caption = True
    mae_mode = "all" if use_pre_conv_layer else "mean"

    mask_ratio = 0.0
    vae_params = CONV_VAE_PARAMS()  #CONV_VAE_BIG_PARAMS() #VAE_PARAMS()
    vae_params.embed_dim = embed_dim
    vae_params.use_epsilon = use_epsilon
    vae_params.use_pre_conv_layer = use_pre_conv_layer

    latent_reconstructor_params = LATENT_RECONSTRUCTOR_PARAMS()


class CAPTIONLESS_LCMVAE_PARAMS:
    checkpoint_file = "captionless_lcmvae"
    embed_dim = 768
    im_dims = [3, 224, 224]


class STANDALONE_VAE_PARAMS:
    checkpoint_file = "standalone_vae"
    im_dims = [3, 224, 224]
    embed_dim = 768
    use_linear_decoder = True
    # use_prev_conv_layer = True
    use_epsilon = True
    use_prev_conv_layer = False
    mask_type = 'None' #'Patch' 'Pixel'
    mask_ratio = 0.0

    encoder_params = LINEAR_NETWORK_PARAMS()
    encoder_params.output_dim = embed_dim * 2
    encoder_params.activation = nn.LeakyReLU()
    encoder_params.linear_layer_params = [
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 768},
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
        {"in_dim": 1536, "out_dim": np.prod(im_dims)}
    ]

    



class CONV_DECODER_512_PARAMS:
    checkpoint_file = "conv_decoder_512"
    embed_dim = 768
    out_channels = 10


