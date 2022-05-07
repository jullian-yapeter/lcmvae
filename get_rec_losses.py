import sys, os, inspect, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyCocoCaption, MyCocoCaptionDetection

from utils import denormalize_torch_to_cv2, count_parameters, has_internet, show_masked_image

from tqdm import tqdm
import glob, importlib


device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
class VAL_DATASET_PARAMS:
    data_root = './data'
    dataType = 'val2017'  # dataType: 'train2017' or 'val2017'
    image_dir = f'{data_root}/coco/{dataType}/'
    det_ann_file = f'{data_root}/coco/ann_trainval2017/instances_{dataType}.json'
    ann_file = f'{data_root}/coco/ann_trainval2017/captions_{dataType}.json'
    transform = None
    from_pretrained = 'facebook/vit-mae-base' \
        if has_internet() else './saved_models/ViTMAE'
    
    # DataLoader
    batch_size = 1
    shuffle = True
    num_workers = 0
    # WARN: when n_workers > 0, DataLoader will work slowly due to unknow reasons.
    
val2017 = MyCocoCaptionDetection(root=VAL_DATASET_PARAMS.image_dir,
                              annFile=VAL_DATASET_PARAMS.ann_file,
                              detAnnFile=VAL_DATASET_PARAMS.det_ann_file,
                              superclasses=["person", "vehicle"],
                              from_pretrained=VAL_DATASET_PARAMS.from_pretrained)

# image mean and std for reconstruction
image_mean = torch.tensor(val2017.feature_extractor.image_mean)
image_std = torch.tensor(val2017.feature_extractor.image_std)


# Check the info of dataset, you can ignore this part
print('-'*40)
val2017.coco.info()
print('-'*40)
print(f'The number of samples: {len(val2017)}')
first_img, (first_cap, first_segment) = val2017[0]
print(f'Image shape: {first_img.size()}')

# Build Dataloader for pretrain
val_data_loader = DataLoader(dataset = val2017, 
                         batch_size=VAL_DATASET_PARAMS.batch_size,
                         shuffle=VAL_DATASET_PARAMS.shuffle, 
                         num_workers=VAL_DATASET_PARAMS.num_workers)
val_data_loader = [next(iter(val_data_loader))]

def get_rec_example(experiment_name, target_id = 0, validating=True, mask_ratio=0): 
    lcmvae = torch.load(
            f"saved_models/{experiment_name}/lcmvae_{experiment_name+'_pretrain'}").eval()
    lcmvae.im_cap_encoder.vit.model.config.mask_ratio = mask_ratio

    im, (cap, _) = val2017[target_id] # if validating else coco2017[target_id]
    im = im.to(device)
    target = denormalize_torch_to_cv2(im, image_mean, image_std)
    # cv2.imwrite(f"{save_dir}/{fix_names.get(experiment_name, experiment_name)}_target.jpg", target)
    reconstruction, mask = lcmvae.run(im[None], [cap])
    prediction = denormalize_torch_to_cv2(reconstruction, image_mean, image_std)
    if not validating:
        target_id = f"T{target_id}"
    white_stripe = torch.zeros(224,2,3) + 255
    save_path = f"reconstruction_results/{int(mask_ratio*100)}_mask" 
    if mask_ratio>0:
        masked_image = show_masked_image(target, mask=mask, patch_size=16)
        output = np.concatenate((target, white_stripe, masked_image, white_stripe, prediction), axis=1)
    else:
        output = np.concatenate((target, white_stripe, prediction), axis=1) 
    os.makedirs(save_path, exist_ok=True)
    save_path += f"/{target_id}_{fix_names.get(experiment_name, experiment_name)}.jpg"
    
    cv2.imwrite(save_path, output)
    print("Reconstruction result saved:", save_path)
    return output

def get_rec_loss(experiment_name, mask_ratio=0): 
    mod_params = importlib.import_module(f"saved_models.{experiment_name}.params_{experiment_name}")
    lcmvae = torch.load(
            f"saved_models/{experiment_name}/lcmvae_{experiment_name+'_pretrain'}").eval()
    lcmvae.im_cap_encoder.vit.model.config.mask_ratio = mask_ratio
    rec_losses = []
    for im_batch, (cap_batch, seg_batch) in tqdm(val_data_loader, desc=f'validation {experiment_name}', mininterval=60):
        im_batch = im_batch.to(device)
        target = im_batch.clone().detach()
        with torch.no_grad():
            outputs, _ = lcmvae(im_batch, cap_batch)
        losses = lcmvae.loss(
            outputs, target, mod_params.PRETRAIN_PARAMS.beta, mod_params.PRETRAIN_PARAMS.delta)
        rec_losses.append(losses[1].cpu().detach().item())
    mean_loss = np.mean(rec_losses)
    return mean_loss, rec_losses


fix_names = {
    "noVar_noLat_noPreC_noMask_noCap_0430_1112": "noVar_noLat_noPreC_noMask_0430_1112",
    "noVar_noLat_noMask_noCap_0430_1113": "noVar_noLat_noMask_0430_1113",
    "noVar_noPreC_noMask_noCap_0430_1113" : "noVar_noPreC_noMask_0430_1113" 
}


exp_list = list(map(os.path.basename, glob.glob('saved_models/*_0505_*')))
exp_list = [x for x in exp_list if 'seg_' not in x ]
print('Generating rec_losses for:\n  ', exp_list)
print('len: ', len(exp_list))

labels = ['model', 'mask', 'caption', 'variational', 'latent_reg', 'pre_conv' , 'img_id', 'rec_mask_ratio', 'rec_loss']
df_losses = pd.DataFrame(columns=labels)

mask_ratio = float(sys.argv[1])
for mod in exp_list:
    mean_loss, all_losses = get_rec_loss(mod, mask_ratio=mask_ratio)
    print(mod, mean_loss)
    to_add = pd.DataFrame(
        {
        'model': [mod] * len(all_losses),
        'mask': ['No' if 'noMask' in mod else 'Yes'] * len(all_losses),
        'caption': ['No' if 'noCap' in mod else 'Yes'] * len(all_losses),
        'variational': ['No' if 'noVar' in mod else 'Yes'] * len(all_losses),
        'latent_reg': ['No' if 'noLat' in mod else 'Yes'] * len(all_losses),
        'pre_conv': ['No' if 'noPreC' in mod else 'Yes'] * len(all_losses),
        'img_id': list(range(len(all_losses))),
        'rec_mask_ratio': [mask_ratio,] * len(all_losses),
        'rec_loss': all_losses
        }
    )

    df_losses = pd.concat([df_losses, to_add], ignore_index=True)
    df_losses.to_csv(f'reconstruction_results/{int(mask_ratio*100)}_all_rec_losses.csv', index=False)
