from utils import save_checkpoint

import numpy as np
import matplotlib.pyplot as plt
import torch


class PreTrainer():
    def __init__(self, lcmvae, PTP, experiment_name=None):
        self.config = PTP
        self.name = experiment_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lcmvae = lcmvae.train()
        self.opt = torch.optim.Adam(self.lcmvae.parameters(),
                               lr=self.config.learning_rate)
        
    def run(self, data):
        train_it = 0
        best_loss = float('inf')
        total_losses, rec_losses, kl_losses = [], [], []
        for ep in range(self.config.epochs):
            print("Run Epoch {}".format(ep))
            for im_batch, cap_batch in data:
                target_batch = np.array(im_batch)
                self.opt.zero_grad()
                outputs, _ = self.lcmvae(im_batch, cap_batch, pretraining=True)
                target_batch = torch.tensor(
                    target_batch).reshape(-1, 224, 224, 3).type(torch.float)
                total_loss, rec_loss, kl_loss = self.lcmvae.loss(
                    target_batch, outputs, self.config.beta)
                total_loss.backward()
                self.opt.step()
                total_losses.append(total_loss.cpu().detach())
                rec_losses.append(rec_loss.cpu().detach())
                kl_losses.append(kl_loss.cpu().detach())
                new_loss = sum(total_losses[-10:]) / len(total_losses[-10:])
                if new_loss < best_loss:
                    save_checkpoint(self.lcmvae.vae.encoder, name=self.name)
                    save_checkpoint(self.lcmvae.vae.decoder, name=self.name)
                    best_loss = new_loss
                if train_it % 5 == 0:
                    print(
                        f"It {train_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()}"
                    )
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


class Trainer():
    def __init__(self, lcmvae, head, criterion, TP, experiment_name=None):
        self.config = TP
        self.name = experiment_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lcmvae = lcmvae.train()
        self.head = head.train()
        self.criterion = criterion
        self.opt = torch.optim.Adam(self.lcmvae.parameters(),
                                    lr=self.config.learning_rate)

    def run(self, data):
        train_it = 0
        best_loss = float('inf')
        losses = []
        for ep in range(self.config.epochs):
            print("Run Epoch {}".format(ep))
            for im_batch, cap_batch, target_batch in data:
                self.opt.zero_grad()
                lcmvae_outputs, _ = self.lcmvae(im_batch, cap_batch, pretraining=False)
                head_outputs = self.head(lcmvae_outputs)
                loss = self.criterion(torch.tensor(
                    target_batch).reshape(-1, 224, 224, 3).type(torch.float), head_outputs)
                loss.backward()
                self.opt.step()
                losses.append(loss.cpu().detach())
                new_loss = sum(losses[-10:]) / len(losses[-10:])
                if new_loss < best_loss:
                    save_checkpoint(self.lcmvae.vae.encoder, name=self.name)
                    save_checkpoint(self.head, name=self.name)
                    best_loss = new_loss
                if train_it % 5 == 0:
                    print(
                        f"It {train_it}: Total Loss: {loss.cpu().detach()}"
                    )
                train_it += 1
        print("Done!")

        # log the loss training curves
        fig = plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(111)
        ax1.plot(losses)
        ax1.title.set_text("Head Loss")
        plt.show()



######################################################
###################  NOT YET TESTED ##################
######################################################
class GeneralTrainer():
    def __init__(self, model, config, criterion, mask_maker=None, feat_extractor=None, experiment_name=None):
        self.config = config
        self.name = experiment_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.train()
        self.feat_extractor = feat_extractor
        self.mask_maker = mask_maker
        self.criterion = criterion
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=self.config.learning_rate)

    def run(self, data):
        train_it = 0
        best_loss = float('inf')
        losses = []

        for ep in range(self.config.epochs):
            print("Run Epoch {}".format(ep))
            for img_batch, cap_batch in data:
                img_batch = img_batch.to(self.device)
                target_batch = img_batch
                if self.mask_maker: 
                    img_batch, masks = self.mask_maker(img_batch)
                if self.feat_extractor:
                    img_batch = self.feat_extractor(img_batch)
                print(img_batch.shape)

                self.opt.zero_grad()
                outputs, _ = self.model(img_batch, cap_batch)
                loss = self.criterion(
                    torch.tensor(target_batch, dtype=torch.float).reshape(-1, 3, 224, 224),
                    outputs)
                loss.backward()
                self.opt.step()
                
                losses.append(loss.cpu().detach())
                new_loss = sum(losses[-10:]) / len(losses[-10:])
                if new_loss < best_loss:
                    save_checkpoint(self.model, name=self.name)
                    best_loss = new_loss
                if train_it % 10 == 0:
                    print(
                        f"It {train_it}: Total Loss: {loss.cpu().detach()}"
                    )
                train_it += 1

        print("Done!")
        # log the loss training curves
        fig = plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(111)
        ax1.plot(losses)
        ax1.title.set_text("Head Loss")
        plt.show()