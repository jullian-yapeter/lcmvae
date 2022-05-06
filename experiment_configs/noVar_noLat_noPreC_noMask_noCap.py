from models.basic_models.params import LINEAR_NETWORK_PARAMS, DECODER_PARAMS
import math, torch.nn as nn
from utils import has_internet

class PRETRAIN_PARAMS:
    epochs = 80
    learning_rate = 2e-4
    beta = 0
    delta = 5e4

class TRAIN_PARAMS:
    epochs = 100
    learning_rate = 2e-4
    beta = 0


class PRETEST_PARAMS:
    beta = 0


class TEST_PARAMS:
    beta = 0
    

class PRETRAIN_DATASET_PARAMS:
    data_root = './data'
    dataType = 'train2017'  # dataType: 'train2017' or 'val2017'
    image_dir = f'{data_root}/coco/{dataType}/'
    det_ann_file = f'{data_root}/coco/ann_trainval2017/instances_{dataType}.json'
    ann_file = f'{data_root}/coco/ann_trainval2017/captions_{dataType}.json'
    transform = None
    # NOTE: set proper from_pretrained for dataset
    # VitEncoder: "google/vit-base-patch16-224-in21k"
    # VitEncoder: 'facebook/vit-mae-base'
    from_pretrained = 'facebook/vit-mae-base' \
        if has_internet() else './saved_models/ViTMAE'
    
    # DataLoader
    batch_size = 256
    shuffle = True
    num_workers = 0
    # WARN: when n_workers > 0, DataLoader will work slowly due to unknow reasons.
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
    use_epsilon = False
    use_pre_conv_layer = False
    is_mae = True
    use_caption = False
    mae_mode = "all" if use_pre_conv_layer else "mean"

    mask_ratio = 0.0
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