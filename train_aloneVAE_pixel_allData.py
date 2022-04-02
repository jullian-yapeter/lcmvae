import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyCocoCaption
from utils import has_internet
from datetime import date

today = date.today()

class PRETRAIN_DATASET_PARAMS:
    data_root = './data'
    dataType = 'train2017'        # dataType: 'train2017' or 'val2017'
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
from models.params import LCMVAE_PARAMS as LCMVAEP
from train import VAEPreTrainer
from models.standalone_vae import StandaloneVAE
from masks import PixelMask, PatchMask

experiment_name = 'alone_VAE_pixelMask'  + today.strftime("-%Y-%m-%d") 
print('-'*40); print("Experiment", experiment_name); print('-'*40)

class PRETRAIN_PARAMS:
    epochs = 10
    learning_rate = 1e-4
    beta = 1e-7


class STANDALONE_VAE_PARAMS:
    checkpoint_file = 'standalone_vae'
    embed_dim = 768
    im_dims = [3, 224, 224]


model = StandaloneVAE(STANDALONE_VAE_PARAMS, device=device)
pretrainer = VAEPreTrainer(
    model, PRETRAIN_PARAMS, mask_maker=PixelMask(0.2),
    experiment_name=experiment_name)
pretrainer.run(data=data_loader)