from models.lcmvae import LCMVAE
from models.params import LCMVAE_PARAMS as LCMVAEP
from train import Trainer
from test import Tester
from params import TRAIN_PARAMS as TP
from params import TEST_PARAMS as TEP
from utils import load_checkpoint

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

def main():
    experiment_name = "test1"
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    # transform = transforms.ToTensor()

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
    trainer = Trainer(lcmvae, TP, experiment_name=experiment_name)
    trainer.run(data)

    test_data = [[dog_im, dog_cap], [cat_im, cat_cap]]
    load_checkpoint(lcmvae, name=experiment_name)
    tester = Tester(lcmvae, TEP, experiment_name=experiment_name)
    tester.run(test_data)

    lcmvae.run([dog_im], [dog_cap], f"output/{experiment_name}.jpg")


if __name__=="__main__":
    main()
