from models.basic_models.params import LINEAR_NETWORK_PARAMS, DECODER_PARAMS
import math, torch.nn as nn
from utils import has_internet


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
        {"in_dim": 1536, "out_dim": 768},
        {"in_dim": 768, "out_dim": encoder_params.output_dim}
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


class LCMVAE_PARAMS:
    checkpoint_file = "lcmvae"
    embed_dim = 768
    use_latent_regularizer = False
    use_epsilon = True
    use_pre_conv_layer = True
    is_mae = True
    use_caption = True
    mae_mode = "all" if use_pre_conv_layer else "mean"

    mask_ratio = 0.75
    vae_params = CONV_VAE_PARAMS()  #CONV_VAE_BIG_PARAMS() #VAE_PARAMS()
    vae_params.embed_dim = embed_dim
    vae_params.use_epsilon = use_epsilon
    vae_params.use_pre_conv_layer = use_pre_conv_layer

    latent_reconstructor_params = LATENT_RECONSTRUCTOR_PARAMS()


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

class STANDALONE_VAE_PARAMS:
    checkpoint_file = "standalone_vae"
    embed_dim = 768
    im_dims = [3, 224, 224]