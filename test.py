from utils import log_losses

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


class Tester():
    def __init__(self, lcmvae, TEP, experiment_name=None):
        self.config = TEP
        self.name = experiment_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lcmvae = lcmvae
        if self.lcmvae.config.is_mae:
            self.lcmvae.config.mask_ratio = 0
        self.lcmvae = self.lcmvae.eval()

    def run(self, data):
        test_it = 0
        total_losses, rec_losses, kl_losses = [], [], []
        for i, (im, cap) in enumerate(data):
            target = np.array(im)
            outputs = self.lcmvae.reconstruct(im, cap)
            # cv2.imwrite(f"output/{i}.jpg", outputs["reconstruction"][0].detach().numpy() * 255)
            target = torch.tensor(
                target).reshape(-1, 224, 224, 3).type(torch.float)
            total_loss, rec_loss, kl_loss = self.lcmvae.loss(
                target, outputs, self.config.beta)
            total_losses.append(total_loss.cpu().detach().item())
            rec_losses.append(rec_loss.cpu().detach().item())
            kl_losses.append(kl_loss.cpu().detach().item())
            print(f"It {test_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()}")
        print("Done!")
        log_losses(total_losses, rec_losses, kl_losses, name=self.name)

