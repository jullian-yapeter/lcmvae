from models.basic_models.linear import Encoder, Decoder
from models.lcmvae import LCMVAE #, LCMVAEDownstream
from models.heads import ReconstructionHead
from models.params import LCMVAE_PARAMS as LCMVAEP
# from models.params import LCMVAED_PARAMS as LCMVAEDP
from train import PreTrainer, Trainer
from test import PreTester, Tester
from params import PRETRAIN_PARAMS as PTP
from params import PRETEST_PARAMS as PTEP
from params import TRAIN_PARAMS as TP
from params import TEST_PARAMS as TEP
from utils import load_checkpoint
from params import PRETRAIN_DATASET_PARAMS

import cv2
import torch
import torch.nn as nn

from PIL import Image
import numpy as np

from torch.utils.data import DataLoader
from dataset import MyCocoCaption


def main():
    experiment_name = "no_mask"
    pretrain = True
    pretest = False
    train = False
    test = False


    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    # dog_url = 'data/coco/val2017/000000000139.jpg'
    # cat_url = 'data/coco/val2017/000000000285.jpg'
    
    # dog_im = cv2.imread(dog_url)
    # cat_im = cv2.imread(cat_url)
    # dog_im = cv2.resize(dog_im, (224, 224))
    # cat_im = cv2.resize(cat_im, (224, 224))
    # images = [dog_im, cat_im]
    # dog_cap = "smiling happy dog"
    # cat_cap = "confused orange cat"
    # captions = [dog_cap, cat_cap]
    # data = [[images, captions]]
    
    # Construct Dataset
    coco_val2017 = MyCocoCaption(root = PRETRAIN_DATASET_PARAMS.image_dir,
                                annFile = PRETRAIN_DATASET_PARAMS.ann_file,
                                from_pretrained = PRETRAIN_DATASET_PARAMS.from_pretrained)

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

    if train:
        test_data = [[dog_im, dog_cap, dog_im], [cat_im, cat_cap, cat_im]]
        head = ReconstructionHead(
            LCMVAEP.vae_params.decoder_params, im_dims=(224, 224, 3))
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
            LCMVAEP.vae_params.decoder_params, im_dims=(224, 224, 3))
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
