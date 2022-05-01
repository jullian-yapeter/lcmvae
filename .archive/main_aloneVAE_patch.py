from models.params import LCMVAE_PARAMS as LCMVAEP, VAE_PARAMS as VAEP
from models.standalone_vae import StandaloneVAE
from models.params import CONV_DECODER_512_PARAMS as CD512P
from models.params import STANDALONE_VAE_PARAMS as SVAEP
from train import Trainer, VAEPreTrainer
from test import Tester
from params import PRETRAIN_PARAMS as PTP
from params import PRETEST_PARAMS as PTEP
from params import TRAIN_PARAMS as TP
from params import TEST_PARAMS as TEP
from utils import load_checkpoint, denormalize_torch_to_cv2
from params import PRETRAIN_DATASET_PARAMS

import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import numpy as np

from torch.utils.data import DataLoader
from dataset import MyCocoCaption, MyCocoCaptionDetection
import sys, os, inspect, time

def main():
    experiment_name = sys.argv[0][5:-3] + time.strftime("_%m%d_%H%M")
    print('-'*40); print("Experiment: ", experiment_name); print('-'*40)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    save_dir = f"./saved_models/{experiment_name}"
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
    with open(f"{save_dir}/params_{experiment_name}.py", 'w+') as f:
        f.write(f"# PARAMS for Experiment: {experiment_name}\n")
        f.write(f"# GPU Type: {torch.cuda.get_device_name()}\n\n")
        f.write(
            "from models.basic_models.params import LINEAR_NETWORK_PARAMS, DECODER_PARAMS\n"
            "from utils import has_internet\n"
            "import math, torch, torch.nn as nn\n\n")
        lines = map(inspect.getsource, [
            PTP, PTEP, TP, TEP, SVAEP, CD512P, PRETRAIN_DATASET_PARAMS])
        f.write('\n\n'.join(lines))

    # detection dataset: outputs: img, (caption, mask)
    # cats = {1: 'person', 2: 'bicycle', 3: 'car',4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat'}
    coco_val2017 = MyCocoCaptionDetection(root=PRETRAIN_DATASET_PARAMS.image_dir,
                                  annFile=PRETRAIN_DATASET_PARAMS.ann_file,
                                  detAnnFile=PRETRAIN_DATASET_PARAMS.det_ann_file,
                                  superclasses=["person", "vehicle"],
                                  from_pretrained=PRETRAIN_DATASET_PARAMS.from_pretrained)

    # image mean and std for reconstruction
    image_mean = torch.tensor(coco_val2017.feature_extractor.image_mean)
    image_std = torch.tensor(coco_val2017.feature_extractor.image_std)

    # Check the info of dataset, you can ignore this part
    print('-'*40)
    coco_val2017.coco.info()
    print('-'*40)
    print(f'The number of samples: {len(coco_val2017)}')
    first_img, (first_cap, first_segment) = coco_val2017[0]
    print(f'Image shape: {first_img.size()}')
    
    # Build Dataloader for pretrain
    data_loader = DataLoader(dataset = coco_val2017, 
                             batch_size=PRETRAIN_DATASET_PARAMS.batch_size, 
                             shuffle=PRETRAIN_DATASET_PARAMS.shuffle, 
                             num_workers=PRETRAIN_DATASET_PARAMS.num_workers)


    # data_loader = [next(iter(data_loader))] # for testing only
    
    # lcmvae = LCMVAE(LCMVAEP, device=device)
    from masks import PatchMask
    mask_maker = PatchMask(0.25, 16)
    svae = StandaloneVAE(SVAEP, device=device)
    pretrainer = VAEPreTrainer(svae, PTP, experiment_name = experiment_name+"_pretrain", mask_maker = mask_maker, save_dir=save_dir)
    pretrainer.run(data=data_loader)


if __name__=="__main__":
    main()


