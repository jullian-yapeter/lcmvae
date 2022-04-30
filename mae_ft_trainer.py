from typing import Iterable, Optional
from tqdm import tqdm

from transformers import ViTMAEForPreTraining

import torch

import matplotlib.pyplot as plt


class MaeFtTrainer:
    """Finetune and evaluate a Masked Autoencoder"""
    
    def __init__(self, model: ViTMAEForPreTraining, 
                #  criterion: torch.nn.Module, # ViTMAEForPreTraining includes loss: MSE
                is_encoder_frozen: bool, is_decoder_frozen: bool, 
                train_data_loader: Optional[Iterable], 
                eval_data_loader: Optional[Iterable], 
                optimizer: torch.optim.Optimizer,
                device: torch.device, epochs: int, verbose: bool = True, 
                is_with_caption: bool = False, args = None):
        
        self.model = model
        self.is_encoder_frozen = is_encoder_frozen
        self.is_decoder_frozen = is_decoder_frozen
        # self.criterion = criterion
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.verbose = verbose
        self.is_with_caption = is_with_caption
        self.args = args
        
        self.encoder = model.vit
        self.decoder = model.decoder
        
        self.model = self.model.to(self.device)
        
        if self.is_encoder_frozen and self.is_decoder_frozen:
                raise ValueError("Both MAE's encoder and decoder are frozen. The model can't be trained")
    
    def train(self):
        train_iter = 0
        best_loss = float('inf')
        rec_losses = []
        
        self.model.train(True)
        
        for ep in tqdm(range(self.epochs), desc = 'Finetune MAE epoch'):
            # freeze encoder or decoder
            if self.is_encoder_frozen:
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False 
                if self.verbose:
                    print("Frozen encoder")
                    # for name, param in model.vit.named_parameters():
                    #     print(name, param.requires_grad)
                        
            if self.is_decoder_frozen:
                for name, param in self.decoder.named_parameters():
                    param.requires_grad = False 
                if self.verbose:
                    print("Frozen decoder")
                    # for name, param in model.decoder.named_parameters():
                    #     print(name, param.requires_grad)
            
            if self.train_data_loader == None:
                print(f"No data loader for training, please set `train_data_loader`")
            
            # iterate each epoch       
            for imgs, _ in self.train_data_loader:
                # one batch of images
                imgs = imgs.to(self.device)
                self.optimizer.zero_grad()
                
                # fit model and predict
                img_outputs = self.model(imgs)  # [batch_size, 196, 786]
                loss = img_outputs.loss  # reconstruction loss
                mask = img_outputs.mask  # patch mask
                ids_restore = img_outputs.ids_restore  # patch restore id
                rec_losses.append(loss.cpu().detach())
                
                # calcualte grad and update params
                loss.backward()
                self.optimizer.step()
                
                if self.verbose:
                    if train_iter % 100 == 0:
                        print(f"Iter {train_iter}: \t Rec Loss: {loss.cpu().detach()}")
                
                train_iter += 1
            
            best_loss = min(rec_losses)
            if self.verbose:
                print(f"Best rec_loss so far: \t {best_loss}")
            
        # plot the rec losses
        
        plt.figure(figsize=(8, 5))
        plt.plot(rec_losses)
        plt.title("Reconstruction Loss (Finetune MAE)")
        name = "MAE_ft_train_rec_loss"
        plt.savefig(f"output/{name}.jpg")
        
        return rec_losses
   
    def evaluate(self):
        eval_iter = 0
        best_loss = float('inf')
        rec_losses = []
        
        self.model.eval()
        
        if self.eval_data_loader == None:
            print(f"No data loader for evaluation, please set `eval_data_loader`")
               
        for imgs, _ in tqdm(self.eval_data_loader, desc = 'Evaluate finetuned MAE iter'):
            # one batch of images
            imgs = imgs.to(self.device)
            
            img_outputs = self.model(imgs)  # [batch_size, 196, 786]
            loss = img_outputs.loss  # reconstruction loss
            mask = img_outputs.mask  # patch mask
            ids_restore = img_outputs.ids_restore  # patch restore id
            rec_losses.append(loss.cpu().detach())
            
            if self.verbose:
                if eval_iter % 10 == 0:
                    print(f"Iter {eval_iter}: \t Rec Loss: {loss.cpu().detach()}")
            
            eval_iter += 1
        
        best_loss = min(rec_losses)
        if self.verbose:
            print(f"Best rec_loss: \t {best_loss}")
            
        # plot the rec losses
        plt.figure(figsize=(8, 5))
        plt.plot(rec_losses)
        plt.title("Reconstruction Loss (Evaluate)")
        name = "MAE_ft_evl_rec_loss"
        plt.savefig(f"output/{name}.jpg")
        
        return rec_losses
        
if __name__ == "__main__":
    # load coco dataset
    coco_val2017 = MyCocoCaption(root = PRETRAIN_DATASET_PARAMS.image_dir,
                            annFile = PRETRAIN_DATASET_PARAMS.ann_file,
                            from_pretrained = PRETRAIN_DATASET_PARAMS.from_pretrained)
    
    # small dataset and dataloader
    samll_len = 512
    coco_val2017_samll = torch.utils.data.Subset(coco_val2017, 
                                                list(range(0, samll_len)))

    data_loader_small = DataLoader(dataset = coco_val2017_samll, 
                            batch_size=PRETRAIN_DATASET_PARAMS.batch_size, 
                            shuffle=PRETRAIN_DATASET_PARAMS.shuffle, 
                            num_workers=PRETRAIN_DATASET_PARAMS.num_workers)
    
    from transformers import AutoFeatureExtractor, ViTMAEForPreTraining
    from PIL import Image

    mae_ft = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

    model = mae_ft  # MaeFtTrainer will send model to GPU when init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 1
    learning_rate = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    # ViTMAEForPreTraining includes loss func: MSE, so no need to build a criterion
    # criterion = nn.MSELoss() 

    verbose = True  # output rec_loss or not

    # fine tune end to end
    is_encoder_frozen = False
    is_decoder_frozen = False

    # # only fine tune MAE decoder
    # is_encoder_frozen = True
    # is_decoder_frozen = False

    mae_ft_trainer = MaeFtTrainer(
        model=model, 
        is_encoder_frozen=is_encoder_frozen, 
        is_decoder_frozen=is_decoder_frozen, 
        train_data_loader=data_loader_small, 
        eval_data_loader=data_loader_small,
        optimizer=optimizer,
        device=device, epochs=epochs, 
        verbose = True, is_with_caption = False, args = None
    )

    train_rec_losses = mae_ft_trainer.train()
    eval_rec_losses = mae_ft_trainer.evaluate()
   