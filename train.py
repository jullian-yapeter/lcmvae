import imp
from utils import save_checkpoint

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


class Trainer():
    def __init__(self, lcmvae, PTP, experiment_name=None, downstream_criterion=None):
        self.config = PTP
        self.name = experiment_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lcmvae = lcmvae.train()
        self.opt = torch.optim.Adam(self.lcmvae.parameters(),
                               lr=self.config.learning_rate)
        self.downstream_criterion = downstream_criterion
        
    def run(self, data):
        train_it = 0
        best_loss = float('inf')
        total_losses = []
        if not self.downstream_criterion:
            rec_losses, kl_losses = [], []
        for ep in range(self.config.epochs):
            print("Run Epoch {}".format(ep))
            batch_i = 0
            for im_batch, (cap_batch, seg_batch) in tqdm(data, desc= f"batch_{batch_i}"):
                # create a batch with 2 images for testing code -> (2, 224, 224, 3)
                # target_batch = np.array(im_batch)
                if self.downstream_criterion:
                    target = seg_batch.clone().detach().squeeze()
                else:
                    target = im_batch.clone().detach()
                self.opt.zero_grad()
                outputs, _ = self.lcmvae(im_batch, cap_batch)
                if self.downstream_criterion:
                    total_loss = self.downstream_criterion(
                        outputs["reconstruction"], target) / target.shape[0]
                else:
                    total_loss, rec_loss, kl_loss = self.lcmvae.loss(
                        outputs, target, self.config.beta)
                total_loss.backward()
                self.opt.step()
                total_losses.append(total_loss.cpu().detach())
                if not self.downstream_criterion:
                    rec_losses.append(rec_loss.cpu().detach())
                    kl_losses.append(kl_loss.cpu().detach())
                new_loss = sum(total_losses[-10:]) / len(total_losses[-10:])
                if new_loss < best_loss:
                    save_checkpoint(self.lcmvae.vae.encoder, name=self.name)
                    save_checkpoint(self.lcmvae.vae.decoder, name=self.name)
                    best_loss = new_loss
                if train_it % 10 == 0:
                    if self.downstream_criterion:
                        print(
                            f"It {train_it}: Total Loss: {total_loss.cpu().detach()}"
                        )
                    else:
                        print(
                            f"It {train_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()}"
                        )
                train_it += 1
                batch_i += 1
        print("Done!")

        # log the loss training curves
        fig = plt.figure(figsize=(15, 5))
        if self.downstream_criterion:
            ax1 = plt.subplot(111)
            ax1.plot(total_losses)
            ax1.title.set_text("Total Loss")
        else:
            ax1 = plt.subplot(131)
            ax1.plot(total_losses)
            ax1.title.set_text("Total Loss")
            ax2 = plt.subplot(132)
            ax2.plot(rec_losses)
            ax2.title.set_text("Reconstruction Loss")
            ax3 = plt.subplot(133)
            ax3.plot(kl_losses)
            ax3.title.set_text("KL Loss")
        plt.show()


# class Trainer():
#     def __init__(self, lcmvae, head, criterion, TP, experiment_name=None):
#         self.config = TP
#         self.name = experiment_name
#         self.device = torch.device(
#             'cuda' if torch.cuda.is_available() else 'cpu')
#         self.lcmvae = lcmvae.train()
#         self.head = head.train()
#         self.criterion = criterion
#         self.opt = torch.optim.Adam(self.lcmvae.parameters(),
#                                     lr=self.config.learning_rate)

#     def run(self, data):
#         train_it = 0
#         best_loss = float('inf')
#         losses = []
#         for ep in range(self.config.epochs):
#             print("Run Epoch {}".format(ep))
#             for im_batch, cap_batch, target_batch in data:
#                 self.opt.zero_grad()
#                 lcmvae_outputs, _ = self.lcmvae(im_batch, cap_batch, pretraining=False)
#                 head_outputs = self.head(lcmvae_outputs)
#                 loss = self.criterion(torch.tensor(
#                     target_batch).reshape(-1, 224, 224, 3).type(torch.float), head_outputs)
#                 loss.backward()
#                 self.opt.step()
#                 losses.append(loss.cpu().detach())
#                 new_loss = sum(losses[-10:]) / len(losses[-10:])
#                 if new_loss < best_loss:
#                     save_checkpoint(self.lcmvae.vae.encoder, name=self.name)
#                     save_checkpoint(self.head, name=self.name)
#                     best_loss = new_loss
#                 if train_it % 5 == 0:
#                     print(
#                         f"It {train_it}: Total Loss: {loss.cpu().detach()}"
#                     )
#                 train_it += 1
#         print("Done!")

#         # log the loss training curves
#         fig = plt.figure(figsize=(10, 5))
#         ax1 = plt.subplot(111)
#         ax1.plot(losses)
#         ax1.title.set_text("Head Loss")
#         plt.show()



class VAEPreTrainer():
    def __init__(self, model, config, mask_maker=None, experiment_name=None):
        self.config = config
        self.name = experiment_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device).train()
        self.mask_maker = mask_maker
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=self.config.learning_rate)

    def run(self, data):
        train_it = 0
        best_loss = float('inf')
        total_losses, rec_losses, kl_losses = [], [], []
        for ep in range(self.config.epochs):
            print("Run Epoch {}".format(ep))
            batch_i = 0
            for im_batch, cap_batch in tqdm(data, desc= f"batch_{batch_i}"):
                im_batch = im_batch.to(self.device)
                if self.mask_maker: 
                    im_batch, masks = self.mask_maker(im_batch)
                self.opt.zero_grad()
                outputs = self.model(im_batch, cap_batch)
                total_loss, rec_loss, kl_loss = self.model.loss(
                    im_batch, outputs, self.config.beta)
                total_loss.backward()
                self.opt.step()

                total_losses.append(total_loss.cpu().detach())
                rec_losses.append(rec_loss.cpu().detach())
                kl_losses.append(kl_loss.cpu().detach())
                new_loss = sum(total_losses[-10:]) / len(total_losses[-10:])
                if new_loss < best_loss:
                    save_checkpoint(self.model, name=self.name)
                    best_loss = new_loss
                if train_it % 10 == 0:
                    print(
                        f"It {train_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()}"
                    )
                train_it += 1
                batch_i += 1

        # log the loss training curves
        fig = plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(121)
        ax1.plot(rec_losses)
        ax1.title.set_text("Reconstruction Loss")
        ax2 = plt.subplot(122)
        ax2.plot(kl_losses)
        ax2.title.set_text("KL Loss")
        plt.show()
