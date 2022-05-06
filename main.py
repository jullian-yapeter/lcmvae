import sys, os, inspect, time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyCocoCaption, MyCocoCaptionDetection

from models.basic_models.conv import ConvDecoder768
from models.lcmvae import LCMVAE
from models.standalone_vae import StandAloneVAE
from train import Trainer
from test import Tester


if len(sys.argv) > 1:
    print("Loading params from ", sys.argv[1])
    import importlib
    params_module = sys.argv[1].replace('/', '.')
    params = importlib.import_module(params_module)
    LCMVAEP = params.LCMVAE_PARAMS
    SVAEP = params.STANDALONE_VAE_PARAMS
    CONV_VAE_PARAMS = params.CONV_VAE_PARAMS
    LINEARP = params.LINEAR_NETWORK_PARAMS
    PTP = params.PRETRAIN_PARAMS
    PTEP = params.PRETEST_PARAMS
    TP = params.TRAIN_PARAMS
    TEP = params.TEST_PARAMS
    PRETRAIN_DATASET_PARAMS = params.PRETRAIN_DATASET_PARAMS
else:
    from models.params import LCMVAE_PARAMS as LCMVAEP
    from models.params import STANDALONE_VAE_PARAMS as SVAEP
    from models.params import CONV_VAE_PARAMS
    from models.basic_models.params import LINEAR_NETWORK_PARAMS as LINEARP
    from params import PRETRAIN_PARAMS as PTP
    from params import PRETEST_PARAMS as PTEP
    from params import TRAIN_PARAMS as TP
    from params import TEST_PARAMS as TEP
    from params import PRETRAIN_DATASET_PARAMS
    
from utils import denormalize_torch_to_cv2, count_parameters, show_masked_image
<<<<<<< HEAD
=======

>>>>>>> e1fecd1e375b88ad6f36bc17293604f37a8b1e0f

def main():
    experiment_name = 'lcmvae_large' + time.strftime("_%m%d_%H%M")
    print('-'*40); print("Experiment: ", experiment_name); print('-'*40)
    pretrain = True
    pretest = True
    train = False
    test = False


    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    # # Construct Dataset
    # coco_val2017 = MyCocoCaption(root = PRETRAIN_DATASET_PARAMS.image_dir,
    #                             annFile = PRETRAIN_DATASET_PARAMS.ann_file,
    #                             from_pretrained = PRETRAIN_DATASET_PARAMS.from_pretrained)

    # detection dataset: outputs: img, (caption, mask)
    # cats = {1: 'person', 2: 'bicycle', 3: 'car',4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat'}
    coco_val2017 = MyCocoCaptionDetection(root=PRETRAIN_DATASET_PARAMS.image_dir,
                                  annFile=PRETRAIN_DATASET_PARAMS.ann_file,
                                  detAnnFile=PRETRAIN_DATASET_PARAMS.det_ann_file,
                                  superclasses=["person", "vehicle"],
                                  from_pretrained=PRETRAIN_DATASET_PARAMS.from_pretrained)
    
    samll_len = 8
    coco_val2017_samll = torch.utils.data.Subset(coco_val2017, 
                                             list(range(0, samll_len)))
    
    data_loader_small = DataLoader(dataset = coco_val2017_samll, 
                         batch_size=PRETRAIN_DATASET_PARAMS.batch_size, 
                          shuffle=PRETRAIN_DATASET_PARAMS.shuffle, 
                          num_workers=PRETRAIN_DATASET_PARAMS.num_workers)

    # image mean and std for reconstruction
    image_mean = torch.tensor(coco_val2017.feature_extractor.image_mean)
    image_std = torch.tensor(coco_val2017.feature_extractor.image_std)

    # Check the info of dataset, you can ignore this part
    print('-'*40)
    coco_val2017.coco.info()
    print('-'*40)
    print(f'The number of samples: {len(coco_val2017)}')
    first_img, (first_cap, first_segment) = coco_val2017[0]
    print(f'Image shape: {first_img.size()}')
    
    # Build Dataloader for pretrain
    data_loader = DataLoader(dataset = coco_val2017, 
                             batch_size=PRETRAIN_DATASET_PARAMS.batch_size,
                             shuffle=PRETRAIN_DATASET_PARAMS.shuffle, 
                             num_workers=PRETRAIN_DATASET_PARAMS.num_workers)

    # # Check: print info for each batch
    # i = 0
    # for imgs, (caps, segment) in data_loader:
    #     print(f'batch_{i}')
    #     print(f"Image batch shape: {imgs.size()}")
    #     print(f"Segmentation batch shape: {segment.size()}")
    #     print(f"Caption batch shape: {len(caps)}")
    #     i += 1
    # exit()
    
    lcmvae = LCMVAE(LCMVAEP, device=device)
    # svae = StandaloneVAE(SVAEP, device=device)

    count_parameters(lcmvae)

    if pretrain:
        pretrainer = Trainer(lcmvae, PTP, experiment_name=experiment_name+"_pretrain")
        pretrainer.run(data=data_loader_small)

    if pretest:
        # # For loading modules separately
        # vit_model = torch.load(f"saved_models/vit_model_{experiment_name+'_pretrain'}")
        # bert_model = torch.load(
        #     f"saved_models/bert_model_{experiment_name+'_pretrain'}")
        # encoder = Encoder(LCMVAEP.vae_params.encoder_params)
        # if LCMVAEP.vae_params.use_linear_decoder:
        #     decoder = Decoder(LCMVAEP.vae_params.decoder_params)
        # else:
        #     decoder = ConvDecoder768(LCMVAEP.vae_params.embed_dim)
        # load_checkpoint(encoder, name=experiment_name+"_pretrain")
        # load_checkpoint(decoder, name=experiment_name+"_pretrain")
        # lcmvae.vae.encoder = encoder
        # lcmvae.vae.decoder = decoder
        # lcmvae.im_cap_encoder.vit.model = vit_model
        # lcmvae.im_cap_encoder.bert.model = bert_model

        # lcmvae = torch.load(
        #     f"saved_models/lcmvae_{experiment_name+'_pretrain'}").eval()
        lcmvae = torch.load(
            f"saved_models/lcmvae_{experiment_name+'_pretrain'}").eval()

        tester = Tester(
            lcmvae, PTEP, experiment_name=experiment_name+"_pretest")
        tester.run(data=data_loader_small)

        for i in range(1):
            im, (cap, _) = coco_val2017[i]
            target = denormalize_torch_to_cv2(im, image_mean, image_std)
            reconstruction, mask = lcmvae.run(im[None], [cap])
            # reconstruction = svae.reconstruct(im[None])["reconstruction"]
            masked_image = show_masked_image(target, mask=mask, patch_size=16)
            prediction = denormalize_torch_to_cv2(
                reconstruction, image_mean, image_std)
<<<<<<< HEAD
            # masked_img = torch.randint(0, 255, [224, 224, 3])
            print(mask)
            masked_image = show_masked_image(target, mask=mask, patch_size=16)
=======
>>>>>>> e1fecd1e375b88ad6f36bc17293604f37a8b1e0f
            result = np.concatenate((target, masked_image, prediction), axis=1)
            cv2.imwrite(f"output/{experiment_name}_{i}.jpg", result)

    if train:
        lcmvae = torch.load(
            f"saved_models/lcmvae_{experiment_name+'_pretrain'}")
        lcmvae.im_cap_encoder.vit.model.config.mask_ratio = 0.0
        decoder = ConvDecoder768(lcmvae.config.embed_dim, out_channels=10, device=device)
        lcmvae.vae.decoder = decoder
        criterion = nn.CrossEntropyLoss(reduction="sum")
        trainer = Trainer(lcmvae, TP, experiment_name=experiment_name+"_train", downstream_criterion=criterion)
        trainer.run(data=data_loader_small)

    if test:
        lcmvae = torch.load(
            f"saved_models/lcmvae_{experiment_name+'_train'}")
        lcmvae.im_cap_encoder.vit.model.config.mask_ratio = 0.0
        criterion = nn.CrossEntropyLoss(reduction="sum")

        tester = Tester(
            lcmvae, TEP, experiment_name=experiment_name+"_test", downstream_criterion=criterion)
        tester.run(data=data_loader_small)

        for i in range(10):
            im, (cap, seg) = coco_val2017[i]
            reconstruction, _ = lcmvae.run(im[None], [cap])
            prediction = torch.argmax(reconstruction, dim=0)
            print(f"Actual classes: {torch.unique(seg)}")
            print(f"Predicted classes: {torch.unique(prediction)}")
            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plt.imshow(seg.squeeze(), vmin=0, vmax=9)
            plt.subplot(122)
            plt.imshow(prediction.squeeze(), vmin=0, vmax=9)
            plt.savefig(f"output/{experiment_name}_segmentation.jpg")
  

if __name__=="__main__":
    main()