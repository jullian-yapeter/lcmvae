import sys, os, inspect, time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import importlib


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyCocoCaption, MyCocoCaptionDetection

from models.basic_models.conv import ConvDecoder768
from models.lcmvae import LCMVAE
from models.standalone_vae import StandaloneVAE
from train import Trainer
from test import Tester
from params import PRETRAIN_PARAMS as PTP
from params import PRETEST_PARAMS as PTEP
from params import TRAIN_PARAMS as TP
from params import TEST_PARAMS as TEP
from params import TRAIN_DATASET_PARAMS as DATASET_PARAMS



    
from utils import denormalize_torch_to_cv2, count_parameters

def main():
    experiment_name = sys.argv[1]
    print('-'*40); print("Segmentation Experiment: ", experiment_name); print('-'*40)
    save_dir = f"./saved_models/{experiment_name}"

    print("Loading params from ", sys.argv[1])
    mod_params_module = f"saved_models.{experiment_name}.params_{experiment_name}"
    mod_params_ = importlib.import_module(mod_params_module)
    LCMVAEP = mod_params_.LCMVAE_PARAMS
        
    # detection dataset: outputs: img, (caption, mask)
    # cats = {1: 'person', 2: 'bicycle', 3: 'car',4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat'}
    coco_val2017 = MyCocoCaptionDetection(root=DATASET_PARAMS.image_dir,
                                  annFile=DATASET_PARAMS.ann_file,
                                  detAnnFile=DATASET_PARAMS.det_ann_file,
                                  superclasses=["person", "vehicle"],
                                  from_pretrained=DATASET_PARAMS.from_pretrained)

    # image mean and std for reconstruction
    image_mean = torch.tensor(coco_val2017.feature_extractor.image_mean)
    image_std = torch.tensor(coco_val2017.feature_extractor.image_std)
    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    # Check the info of dataset, you can ignore this part
    print('-'*40)
    coco_val2017.coco.info()
    print('-'*40)
    print(f'The number of samples: {len(coco_val2017)}')
    first_img, (first_cap, first_segment) = coco_val2017[0]
    print(f'Image shape: {first_img.size()}')
    
    # Build Dataloader for pretrain
    data_loader = DataLoader(dataset = coco_val2017, 
                             batch_size=DATASET_PARAMS.batch_size,
                             shuffle=DATASET_PARAMS.shuffle, 
                             num_workers=DATASET_PARAMS.num_workers)


    data_loader = [next(iter(data_loader))] # for testing only
    
    lcmvae = torch.load(
        f"{save_dir}/lcmvae_{experiment_name+'_pretrain'}")
    lcmvae.im_cap_encoder.vit.model.config.mask_ratio = 0.0
    decoder = ConvDecoder768(lcmvae.config.embed_dim, out_channels=10, device=device)
    lcmvae.vae.decoder = decoder
    criterion = nn.CrossEntropyLoss(reduction="sum")
    trainer = Trainer(lcmvae, TP, experiment_name = experiment_name+"_seg", downstream_criterion=criterion, save_dir=save_dir)
    trainer.run(data=data_loader)

if __name__=="__main__":
    main()


