from models.lcmvae import LCMVAE
from models.params import VAE_PARAMS as VAEP
from train import Trainer
from params import TRAIN_PARAMS as TP

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()

    dog_im = cv2.imread("dataset/images/dog.png") / 255
    cat_im = cv2.imread("dataset/images/cat.png") / 255
    dog_im = cv2.resize(dog_im, (224, 224))
    cat_im = cv2.resize(cat_im, (224, 224))
    images = [dog_im, cat_im]
    dog_cap = "smiling happy dog"
    cat_cap = "confused orange cat"
    captions = [dog_cap, cat_cap]
    # data = zip(images, captions)
    data = [[dog_im, dog_cap]]

    lcmvae = LCMVAE(VAEP, device=device)
    trainer = Trainer(lcmvae, TP)
    trainer.run(data)

    reconstruction = lcmvae.reconstruct(dog_im, dog_cap)

    print(f"reconstruction.shape: {reconstruction[0].shape}")
    cv2.imshow("reconstruction", reconstruction[0].cpu().detach().numpy())
    cv2.waitKey(0)


if __name__=="__main__":
    main()
