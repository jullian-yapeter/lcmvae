from models.lcmvae import LCMVAE
from models.params import VAE_PARAMS as VAEP
from params import TRAIN_PARAMS as TP

import matplotlib.pyplot as plt
import torch


class Trainer():
    def __init__(self, lcmvae, TP):
        self.config = TP
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lcmvae = lcmvae
        self.lcmvae = self.lcmvae.train()
        self.opt = torch.optim.Adam(self.lcmvae.parameters(),
                               lr=self.config.learning_rate)
        
    def run(self, data):
        train_it = 0
        rec_losses, kl_losses = [], []
        for ep in range(self.config.epochs):
            print("Run Epoch {}".format(ep))
            for im_batch, cap_batch in data:
                self.opt.zero_grad()
                outputs = self.lcmvae(im_batch, cap_batch)
                target_batch = torch.tensor(
                    im_batch).reshape(-1, 224, 224, 3).type(torch.float)
                total_loss, rec_loss, kl_loss = self.lcmvae.loss(
                    target_batch, outputs, self.config.beta)
                total_loss.backward()
                self.opt.step()
                rec_losses.append(rec_loss.cpu().detach())
                kl_losses.append(kl_loss.cpu().detach())
                if train_it % 5 == 0:
                    print(
                        f"It {train_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()}")
                train_it += 1
        print("Done!")

        # log the loss training curves
        fig = plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(121)
        ax1.plot(rec_losses)
        ax1.title.set_text("Reconstruction Loss")
        ax2 = plt.subplot(122)
        ax2.plot(kl_losses)
        ax2.title.set_text("KL Loss")
        plt.show()
