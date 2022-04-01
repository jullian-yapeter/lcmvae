from models.basic_models.linear import Encoder, Decoder
from models.lcmvae import LCMVAE
from models.heads import ReconstructionHead
from models.params import LCMVAE_PARAMS as LCMVAEP
from train import PreTrainer, Trainer
from test import PreTester, Tester
from params import PRETRAIN_PARAMS as PTP
from params import PRETEST_PARAMS as PTEP
from params import TRAIN_PARAMS as TP
from params import TEST_PARAMS as TEP
from utils import load_checkpoint
from params import PRETRAIN_DATASET_PARAMS, MODE_PARAMS

import cv2
import torch
import torch.nn as nn

from PIL import Image
import numpy as np

from torch.utils.data import DataLoader
from dataset import MyCocoCaption, MyCocoCaptionDetection


def main():
    experiment_name = MODE_PARAMS.experiment_name
    pretrain = MODE_PARAMS.pretrain
    pretest = MODE_PARAMS.pretest
    train = MODE_PARAMS.train
    test = MODE_PARAMS.test
    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Construct Dataset
    coco_val2017 = MyCocoCaption(root = PRETRAIN_DATASET_PARAMS.image_dir,
                                annFile = PRETRAIN_DATASET_PARAMS.ann_file,
                                from_pretrained = PRETRAIN_DATASET_PARAMS.from_pretrained)
    
<<<<<<< HEAD
    # mean and std for reconstruction
=======
    # image mean and std for reconstruction
>>>>>>> 4d32d50a6c7adde26b2e57d2050796c8d56a5f67
    image_mean = coco_val2017.feature_extractor.image_mean
    image_std = coco_val2017.feature_extractor.image_std

    # # detection dataset: outputs: img, (caption, mask)
    # # cats = {1: 'person', 2: 'bicycle', 3: 'car',4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat'}
    # coco_val2017 = MyCocoCaptionDetection(root=PRETRAIN_DATASET_PARAMS.image_dir,
    #                               annFile=PRETRAIN_DATASET_PARAMS.ann_file,
    #                               detAnnFile=PRETRAIN_DATASET_PARAMS.det_ann_file,
    #                               superclasses=["person", "vehicle"],
    #                               from_pretrained=PRETRAIN_DATASET_PARAMS.from_pretrained)

    # Check the info of dataset, you can ignore this part
    print('-'*40)
    coco_val2017.coco.info()
    print('-'*40)
    print(f'The number of samples: {len(coco_val2017)}')
    first_img, first_cap = coco_val2017[0]
    print(f'Image shape: {first_img.size()}')
    
    # Build Dataloader for pretrain
    data_loader = DataLoader(dataset = coco_val2017, 
                             batch_size=PRETRAIN_DATASET_PARAMS.batch_size, 
                             shuffle=PRETRAIN_DATASET_PARAMS.shuffle, 
                             num_workers=PRETRAIN_DATASET_PARAMS.num_workers)

    # Check: print info for each batch
    # i = 0
    # for imgs, caps in data_loader:
    #     print(f'batch_{i}')
    #     print(f"Image batch shape: {imgs.size()}")
    #     print(f"Caption batch shape: {len(caps)}")
    #     i += 1
    # exit()
    
    lcmvae = LCMVAE(LCMVAEP, device=device)
    if pretrain:
        pretrainer = PreTrainer(lcmvae, PTP, experiment_name=experiment_name+"_pretrain")
        pretrainer.run(data=data_loader)

    if pretest:
        encoder = Encoder(LCMVAEP.vae_params.encoder_params)
        decoder = Decoder(LCMVAEP.vae_params.decoder_params)
        load_checkpoint(encoder, name=experiment_name+"_pretrain")
        load_checkpoint(decoder, name=experiment_name+"_pretrain")
        lcmvae.vae.encoder = encoder
        lcmvae.vae.decoder = decoder

        test_data = [[dog_im, dog_cap], [cat_im, cat_cap]]
        tester = PreTester(
            lcmvae, PTEP, experiment_name=experiment_name+"_pretest")
        tester.run(test_data)

        reconstruction, mask = lcmvae.run([dog_im], [dog_cap])
        print(mask)
        cv2.imwrite(f"output/{experiment_name}.jpg", reconstruction)

    
    im_dims = (3,224,224)
    if train:
        test_data = [[dog_im, dog_cap, dog_im], [cat_im, cat_cap, cat_im]]
        head = ReconstructionHead(
            LCMVAEP.vae_params.decoder_params, im_dims=im_dims)
        encoder = Encoder(LCMVAEP.vae_params.encoder_params)
        load_checkpoint(encoder, name=experiment_name+"_pretrain")
        lcmvae.vae.encoder = encoder
        criterion = nn.MSELoss()
        trainer = Trainer(lcmvae, head, criterion, TP,
                          experiment_name=experiment_name+"_train")
        trainer.run(test_data)

    if test:
        test_data = [[dog_im, dog_cap, dog_im], [cat_im, cat_cap, cat_im]]
        head = ReconstructionHead(
            LCMVAEP.vae_params.decoder_params, im_dims=im_dims)
        encoder = Encoder(LCMVAEP.vae_params.encoder_params)
        load_checkpoint(head, name=experiment_name+"_train")
        load_checkpoint(encoder, name=experiment_name+"_train")
        lcmvae.vae.encoder = encoder
        criterion = nn.MSELoss()
        tester = Tester(lcmvae, head, criterion,
                          TEP, experiment_name=experiment_name+"_test")
        tester.run(test_data)

    

if __name__=="__main__":
    main()
