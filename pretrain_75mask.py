import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyCocoCaption
from utils import has_internet
from datetime import date

today = date.today()


class PRETRAIN_DATASET_PARAMS:
    data_root = './data'
    dataType = 'val2017'        # dataType: 'train2017' or 'val2017'
    image_dir = f'{data_root}/coco/{dataType}/'
    det_ann_file = f'{data_root}/coco/ann_trainval2017/instances_{dataType}.json'
    ann_file = f'{data_root}/coco/ann_trainval2017/captions_{dataType}.json'
    transform = None
    from_pretrained = 'facebook/vit-mae-base' \
        if has_internet() else './saved_models/ViTMAE'
    batch_size = 64
    shuffle = False
    num_workers = 0
    
# Construct Dataset
coco_val2017 = MyCocoCaption(root = PRETRAIN_DATASET_PARAMS.image_dir,
                            annFile = PRETRAIN_DATASET_PARAMS.ann_file,
                            from_pretrained = PRETRAIN_DATASET_PARAMS.from_pretrained)

# image mean and std for reconstruction
image_mean = coco_val2017.feature_extractor.image_mean
image_std = coco_val2017.feature_extractor.image_std

# Check the info of dataset, you can ignore this part
print('-'*40)
coco_val2017.coco.info()
print('-'*40)
print(f'The number of samples: {len(coco_val2017)}')
first_img, first_cap = coco_val2017[0]
print(f'Image shape: {first_img.size()}')

print('-'*40)
print("Has GPU? ", torch.cuda.is_available(), 'Type: ', torch.cuda.get_device_name(0))
print('-'*40)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Build Dataloader for pretrain
data_loader = DataLoader(dataset = coco_val2017, 
                            batch_size=PRETRAIN_DATASET_PARAMS.batch_size, 
                            shuffle=PRETRAIN_DATASET_PARAMS.shuffle, 
                            num_workers=PRETRAIN_DATASET_PARAMS.num_workers)


################################################################################
########################     TRAINING BLOCK    #################################
################################################################################
from models.basic_models.params import LINEAR_NETWORK_PARAMS, DECODER_PARAMS
from train import PreTrainer
from models.lcmvae import LCMVAE
import math

experiment_name = '75mask_pretrain'  + today.strftime("-%Y-%m-%d") 
print('-'*40); print("Experiment", experiment_name); print('-'*40)

class VAE_PARAMS:
    checkpoint_file = "vae" + '-' + experiment_name
    embed_dim = 256
    im_dims = [3, 224, 224]

    encoder_params = LINEAR_NETWORK_PARAMS()
    encoder_params.output_dim = embed_dim * 2
    encoder_params.activation = nn.LeakyReLU()
    encoder_params.linear_layer_params = [
        {"in_dim": 1536, "out_dim": 768},
        {"in_dim": 768, "out_dim": 512},
        {"in_dim": 512, "out_dim": 256},
        {"in_dim": 256, "out_dim": 256},
        {"in_dim": 256, "out_dim": encoder_params.output_dim}
    ]

    decoder_params = DECODER_PARAMS()
    decoder_params.im_dims = (3, 224, 224)
    decoder_params.linear_params.output_dim = embed_dim
    decoder_params.linear_params.activation = nn.LeakyReLU()
    decoder_params.linear_params.linear_layer_params = [
        {"in_dim": embed_dim, "out_dim": 256},
        {"in_dim": 256, "out_dim": 256},
        {"in_dim": 256, "out_dim": 256},
        {"in_dim": 256, "out_dim": 512},
        {"in_dim": 512, "out_dim": math.prod(im_dims)}
    ]


class PRETRAIN_PARAMS:
    epochs = 2
    learning_rate = 1e-4
    beta = 1e-7
    
class LCMVAE_PARAMS:
    checkpoint_file = experiment_name
    is_mae = True
    mask_ratio = 0.75
    vae_params = VAE_PARAMS()
    no_caption = False


lcmvae = LCMVAE(LCMVAE_PARAMS, device=device)
pretrainer = PreTrainer(lcmvae, PRETRAIN_PARAMS, experiment_name=experiment_name)
pretrainer.run(data=data_loader)