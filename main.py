from models.frozen_transformers import ImageCaptionEncoder
from models.vae import VAE
from models.params import VAE_PARAMS as VAEP

import cv2
import torch

def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    dog_im = cv2.imread("dataset/images/dog.png")
    cat_im = cv2.imread("dataset/images/cat.png")
    images = [dog_im, cat_im]
    captions = ["smiling happy dog", "confused orange cat"]

    im_cap_encoder = ImageCaptionEncoder(device=device)
    vae = VAE(VAEP, device=device)

    im_cap_embedding = im_cap_encoder.forward(images, captions)
    vae_output = vae(im_cap_embedding)

    cv2.imshow("im1", vae_output[0].cpu().detach().numpy())
    cv2.waitKey(0)


if __name__=="__main__":
    main()
