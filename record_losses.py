from models.basic_models.linear import Encoder, Decoder
from models.lcmvae import LCMVAE
from models.heads import ReconstructionHead
# from models.params import LCMVAE_PARAMS as LCMVAEP
from train import Trainer
from test import Tester
from params import PRETRAIN_PARAMS as PTP
from params import PRETEST_PARAMS as PTEP
from params import TRAIN_PARAMS as TP
from params import TEST_PARAMS as TEP
from utils import load_checkpoint, denormalize_torch_to_cv2
from params import PRETRAIN_DATASET_PARAMS
import os
import cv2
import torch
import torch.nn as nn

from PIL import Image
import numpy as np

from torch.utils.data import DataLoader
from dataset import MyCocoCaption, MyCocoCaptionDetection

from models.standalone_vae import StandaloneVAE
from models.params import STANDALONE_VAE_PARAMS
from matplotlib import pyplot as plt
from datetime import date

import importlib
from cv2 import imshow as cv2_imshow


device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

# detection dataset: outputs: img, (caption, mask)
# cats = {1: 'person', 2: 'bicycle', 3: 'car',4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat'}
coco_val2017 = MyCocoCaptionDetection(root=PRETRAIN_DATASET_PARAMS.image_dir,
                                annFile=PRETRAIN_DATASET_PARAMS.ann_file,
                                detAnnFile=PRETRAIN_DATASET_PARAMS.det_ann_file,
                                superclasses=["person", "vehicle"],
                                from_pretrained=PRETRAIN_DATASET_PARAMS.from_pretrained)
data_loader = DataLoader(dataset = coco_val2017, 
                            batch_size=PRETRAIN_DATASET_PARAMS.batch_size, 
                        #  batch_size=2,
                            shuffle=PRETRAIN_DATASET_PARAMS.shuffle, 
                            num_workers=PRETRAIN_DATASET_PARAMS.num_workers)
# image mean and std for reconstruction
image_mean = torch.tensor(coco_val2017.feature_extractor.image_mean, device=device)
image_std = torch.tensor(coco_val2017.feature_extractor.image_std, device=device)


# reconstruction loss
def loss_rec(name):    
    encoder_file = 'saved_models/{}/{}'.format(name, 'encoder_'+name+'_pretrain')
    decoder_file = 'saved_models/{}/{}'.format(name, 'decoder_'+name+'_pretrain')
    params_path  = 'saved_models.{}.{}'.format(name, 'params_'+name)


    params_ = importlib.import_module(params_path)
    params_svp = params_.SMALL_VAE_PARAMS
    params_lcmave= params_.LCMVAEP
    today = date.today()
    experiment_name = "sample_run" + today.strftime("-%Y-%m-%d") 

    pretrain = False
    pretest = True
    train = False
    test = False


    
    lcmvae = LCMVAE(params_lcmave, device=device)
    # # Construct Dataset
    # coco_val2017 = MyCocoCaption(root = PRETRAIN_DATASET_PARAMS.image_dir,
    #                             annFile = PRETRAIN_DATASET_PARAMS.ann_file,
    #                             from_pretrained = PRETRAIN_DATASET_PARAMS.from_pretrained)

    
    if pretest:
        # lcmvae.im_cap_encoder.vit.model.config.mask_ratio = 0
        encoder = Encoder(params_svp.encoder_params)
        decoder = Decoder(params_svp.decoder_params)
        load_checkpoint(encoder, checkpoint_file= encoder_file, name=experiment_name+"_pretrain")
        load_checkpoint(decoder, checkpoint_file= decoder_file,name=experiment_name+"_pretrain")
        
        lcmvae.vae.encoder = encoder.to(device)
        lcmvae.vae.decoder = decoder.to(device)
        # svae.encoder = encoder
        # svae.decoder = encoder

        tester = Tester(
            lcmvae, PTEP, experiment_name=experiment_name+"_pretest")
        rec_loss = tester.run(data=data_loader)
        return rec_loss



# segmentation loss
from models.heads import ConvDecoder512
def loss_seg(name,flag):
    if flag==True:
      encoder_file = 'saved_models/{}/{}'.format(name, 'encoder_'+name+'_train')
      decoder_file = 'saved_models/{}/{}'.format(name, 'conv_decoder_512_'+name+'_train')
    else:
      encoder_file = 'saved_models/{}/{}'.format(name, 'encoder_'+name+'_test')
      decoder_file = 'saved_models/{}/{}'.format(name, 'conv_decoder_512_'+name+'_test')
    
    params_path  = 'saved_models.{}.{}'.format(name, 'params_'+name)

    # encoder_file = importlib.import_module(encoder_path)
    # decoder_file = importlib.import_module(decoder_path)
    params_ = importlib.import_module(params_path)
    params_svp = params_.SMALL_VAE_PARAMS
    params_lcmave= params_.LCMVAEP
    params_cd = params_.CD512P
    today = date.today()
    experiment_name = "sample_run" + today.strftime("-%Y-%m-%d") 

    pretrain = False
    pretest = False
    train = False
    test = True

    
    lcmvae = LCMVAE(params_lcmave, device=device)

    if test:
        # lcmvae.im_cap_encoder.vit.model.config.mask_ratio = 0
        lcmvae.im_cap_encoder.vit.model.config.mask_ratio = 0
        encoder = Encoder(params_lcmave.vae_params.encoder_params)
        decoder = ConvDecoder512(params_cd)
        load_checkpoint(encoder, checkpoint_file= encoder_file, name=experiment_name+"_pretrain")
        load_checkpoint(decoder, checkpoint_file= decoder_file,name=experiment_name+"_pretrain")
        
        lcmvae.vae.encoder = encoder
        lcmvae.vae.decoder = decoder

        lcmvae.eval()
        # svae.encoder = encoder
        # svae.decoder = encoder

        criterion = nn.CrossEntropyLoss(reduction="sum")
        tester = Tester(
            lcmvae, PTEP, experiment_name=experiment_name+"_test", downstream_criterion=criterion)
        seg_loss = tester.run(data=data_loader)
        return seg_loss

        
        
model_0403 = ['smallAE_0403_0036','smallAE_capless_0403_0040','smallAE_noMask_0403_0036','small_0403_0036','small_capless_0403_0036','small_capless_noMask_0403_0039','small_noMask_0403_0039']
model_0402 = ['small_0402_1804',
'small_capless_0402_1805',
'small_capless_noMask_0402_1812',
'small_noMask_0402_1827',
'smallAE_0402_1805',
'smallAE_capless_0402_1829',
'smallAE_capless_0402_1837',
'smallAE_noMask_0402_1805'
]

model_all = model_0403 + model_0402



rec_loss = []
seg_loss = []
names = []

import pandas as pd
for i, name in enumerate(model_0403):      
    names += [name]
    rec_loss += [loss_rec(name)]
    seg_loss += [loss_seg(name,True)]
    if i != 0:
        df = pd.DataFrame({
            'model': names,
            'rec_loss': rec_loss,
            'seg_loss': seg_loss
            })
        df.to_csv('./losses.csv', index=False, header=True)
        

