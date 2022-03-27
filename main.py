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

import cv2
import torch
import torch.nn as nn


def main():
    experiment_name = "no_mask"
    pretrain = False
    pretest = False
    train = False
    test = True


    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    dog_im = cv2.imread("dataset/images/dog.png") / 255
    cat_im = cv2.imread("dataset/images/cat.png") / 255
    dog_im = cv2.resize(dog_im, (224, 224))
    cat_im = cv2.resize(cat_im, (224, 224))
    images = [dog_im, cat_im]
    dog_cap = "smiling happy dog"
    cat_cap = "confused orange cat"
    captions = [dog_cap, cat_cap]
    data = [[images, captions]]

    lcmvae = LCMVAE(LCMVAEP, device=device)
    if pretrain:
        pretrainer = PreTrainer(lcmvae, PTP, experiment_name=experiment_name+"_pretrain")
        pretrainer.run(data)

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
