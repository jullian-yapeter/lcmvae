from utils import has_internet, save_checkpoint, save_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

HAS_INTERNET = has_internet() 
class Trainer():
    def __init__(self, lcmvae, PTP, experiment_name=None, downstream_criterion=None, save_dir="saved_models"):
        self.save_dir = save_dir
        self.config = PTP
        self.name = experiment_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lcmvae = lcmvae.train()
        self.lcmvae.im_cap_encoder.vit.model.eval()
        if self.lcmvae.im_cap_encoder.bert:
            self.lcmvae.im_cap_encoder.bert.model.eval()
        self.opt = torch.optim.Adam(self.lcmvae.vae.parameters(),
                               lr=self.config.learning_rate)
        self.downstream_criterion = downstream_criterion
        
    def run(self, data):
        train_it = 0
        best_loss = float('inf')
        total_losses, rec_losses, kl_losses, lat_rec_losses = [], [], [], []

        for ep in range(self.config.epochs):
            print(f"Epoch: {ep}\n")
            for im_batch, (cap_batch, seg_batch) in tqdm(
                data, desc=f"Epoch {ep}", mininterval=5 if HAS_INTERNET else 180):
                # create a batch with 2 images for testing code -> (2, 224, 224, 3)
                # target_batch = np.array(im_batch)
                im_batch = im_batch.to(self.device)
                seg_batch = seg_batch.to(self.device)
                if self.downstream_criterion:
                    target = seg_batch.clone().detach().squeeze(dim=1)
                else:
                    target = im_batch.clone().detach()
                self.opt.zero_grad()
                outputs, _ = self.lcmvae(im_batch, cap_batch)
                if self.downstream_criterion:
                    total_loss = self.downstream_criterion(
                        outputs["reconstruction"], target) / target.shape[0]
                elif self.lcmvae.config.use_latent_regularizer:
                    total_loss, rec_loss, kl_loss, lat_rec_loss = self.lcmvae.loss(
                        outputs, target, self.config.beta, delta=self.config.delta)
                else:
                    total_loss, rec_loss, kl_loss = self.lcmvae.loss(
                        outputs, target, self.config.beta)
                total_loss.backward()
                self.opt.step()
                total_losses.append(total_loss.cpu().detach())
                if not self.downstream_criterion:
                    rec_losses.append(rec_loss.cpu().detach())
                    kl_losses.append(kl_loss.cpu().detach())
                    if self.lcmvae.config.use_latent_regularizer:
                        lat_rec_losses.append(lat_rec_loss.cpu().detach())
                if train_it % 5 == 0:
                    new_loss = np.mean(total_losses[-10:])
                    if new_loss < best_loss:
                        # # To save the individual components
                        # save_checkpoint(self.lcmvae.vae.encoder, name=self.name)
                        # save_checkpoint(self.lcmvae.vae.decoder, name=self.name)
                        # if self.lcmvae.config.use_pre_conv_layer:
                        #     save_checkpoint(
                        #         self.lcmvae.vae.im_embed_pre_conv, name=self.name)
                        save_model(self.lcmvae, name=self.name, save_dir=self.save_dir)
                        best_loss = new_loss
                        if self.downstream_criterion:
                            print(
                                f"It {train_it}: Total Loss: {total_loss.cpu().detach()}"
                            )
                        elif self.lcmvae.config.use_latent_regularizer:
                            print(
                                f"It {train_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()},\t LatRec Loss: {lat_rec_loss.cpu().detach()}"
                            )
                        else:
                            print(
                                f"It {train_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()}"
                            )
                if train_it % 100 == 0:
                    # log the loss training curves
                    plt.figure(figsize=(15, 5))
                    if self.downstream_criterion:
                        ax1 = plt.subplot(111)
                        ax1.plot(total_losses)
                        ax1.title.set_text("Total Loss")
                    elif self.lcmvae.config.use_latent_regularizer:
                        fig, ax = plt.subplots(2, 2)
                        ax[0, 0].plot(total_losses)
                        ax[0, 0].title.set_text("Total Loss")
                        ax[0, 1].plot(rec_losses)
                        ax[0, 1].title.set_text("Reconstruction Loss")
                        ax[1, 0].plot(kl_losses)
                        ax[1, 0].title.set_text("KL Loss")
                        ax[1, 1].plot(lat_rec_losses)
                        ax[1, 1].title.set_text("Latent Reconstruction Loss")
                        fig.tight_layout()
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
                    plt.savefig(f"{self.save_dir}/{self.name}_plot.jpg")
                    plt.close('all')

                train_it += 1
                #batch_i += 1
            if self.downstream_criterion:
                print(
                    f"It {train_it}: Total Loss: {total_loss.cpu().detach()}"
                )
            elif self.lcmvae.config.use_latent_regularizer:
                print(
                    f"It {train_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()},\t LatRec Loss: {lat_rec_loss.cpu().detach()}"
                )
            else:
                print(
                    f"It {train_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()}"
                )
        print("Done!")

        # log the loss training curves
        plt.figure(figsize=(15, 5))
        if self.downstream_criterion:
            ax1 = plt.subplot(111)
            ax1.plot(total_losses)
            ax1.title.set_text("Total Loss")
        elif self.lcmvae.config.use_latent_regularizer:
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].plot(total_losses)
            ax[0, 0].title.set_text("Total Loss")
            ax[0, 1].plot(rec_losses)
            ax[0, 1].title.set_text("Reconstruction Loss")
            ax[1, 0].plot(kl_losses)
            ax[1, 0].title.set_text("KL Loss")
            ax[1, 1].plot(lat_rec_losses)
            ax[1, 1].title.set_text("Latent Reconstruction Loss")
            fig.tight_layout()
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
        plt.savefig(f"{self.save_dir}/{self.name}_plot.jpg")
        plt.close('all')
        losses_df = pd.DataFrame({
            'total_loss': pd.Series(total_losses, dtype=np.float64),
            'rec_loss': pd.Series(rec_losses, dtype=np.float64),
            'kl_losses': pd.Series(kl_losses, dtype=np.float64),
            'lat_rec_losses': pd.Series(lat_rec_losses, dtype=np.float64)
            })
        losses_df.to_csv(f"{self.save_dir}/{self.name}_losses.csv", index=False)

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
    def __init__(self, model, config, experiment_name=None, save_dir=None):
        self.save_dir = save_dir
        self.config = config
        self.name = experiment_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device).train()
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=self.config.learning_rate)

    def run(self, data):
        train_it = 0
        best_loss = float('inf')
        total_losses, rec_losses, kl_losses = [], [], []
        for ep in range(self.config.epochs):
            print("Run Epoch {}".format(ep))
            batch_i = 0
            for im_batch, (cap_batch, seg_batch) in tqdm(data, desc=f"batch_{batch_i}", mininterval=10):
                im_batch = im_batch.to(self.device)
                self.opt.zero_grad()
                outputs = self.model.forward(im_batch)
                total_loss, rec_loss, kl_loss = self.model.loss(
                    outputs, im_batch, self.config.beta)
                total_loss.backward()

                self.opt.step()

                total_losses.append(total_loss.cpu().detach())
                rec_losses.append(rec_loss.cpu().detach())
                kl_losses.append(kl_loss.cpu().detach())
                if train_it % 5 == 0:
                    new_loss = sum(total_losses[-10:]) / len(total_losses[-10:])
                    if new_loss < best_loss:
                        save_model(self.model, name=self.name, save_dir=self.save_dir)
                        best_loss = new_loss
                if train_it % 100 == 0:
                          # log the loss training curves
                    plt.figure(figsize=(15, 5))
                    ax1 = plt.subplot(131)
                    ax1.plot(total_losses)
                    ax1.title.set_text("Total Loss")
                    ax2 = plt.subplot(132)
                    ax2.plot(rec_losses)
                    ax2.title.set_text("Reconstruction Loss")
                    ax3 = plt.subplot(133)
                    ax3.plot(kl_losses)
                    ax3.title.set_text("KL Loss")
                    plt.savefig(f"{self.save_dir}/{self.name}_plot.jpg")
                    plt.close('all')
                    print(
                        f"It {train_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()}"
                    )
                train_it += 1

            plt.figure(figsize=(15, 5))
            ax1 = plt.subplot(131)
            ax1.plot(total_losses)
            ax1.title.set_text("Total Loss")
            ax2 = plt.subplot(132)
            ax2.plot(rec_losses)
            ax2.title.set_text("Reconstruction Loss")
            ax3 = plt.subplot(133)
            ax3.plot(kl_losses)
            ax3.title.set_text("KL Loss")
            plt.savefig(f"{self.save_dir}/{self.name}_plot.jpg")
            plt.close('all')
            print(
                f"It {train_it}: Total Loss: {total_loss.cpu().detach()}, \t Rec Loss: {rec_loss.cpu().detach()},\t KL Loss: {kl_loss.cpu().detach()}"
            )
            plt.close('all')

            losses_df = pd.DataFrame({
                'total_loss': pd.Series(total_losses, dtype=np.float64),
                'rec_loss': pd.Series(rec_losses, dtype=np.float64),
                'kl_loss': pd.Series(kl_losses, dtype=np.float64)
                })
            losses_df.to_csv(f"{self.save_dir}/{self.name}_losses.csv", index=False)
            print("Done!")

  

