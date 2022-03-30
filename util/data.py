import os
from PIL import Image
from typing import List

import torch
from torchvision.datasets import CocoDetection, CocoCaptions
from torchvision import transforms as T

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, random_split

try:
  import pycocotools
except:
    print("torchvision.datasets.CocoDetection requires the COCO API to be installed.")
    print(f"COCO API for python: pycocotools. Do you want to install pycocotools right now? [y/n]")
    install = input()
    if install == 'y':
        # WARN: recommand setting env via conda. pip install pycocotools may bring some errors for numpy.
        # https://github.com/scikit-image/scikit-image/issues/5270
        os.system('conda install pycocotools')
        import pycocotools
    else:
        exit()


IMG_W = IMG_H = 224


def show_images(imgs: List[torch.Tensor]):
    """Transform image Tensors to PIL.Image and then show them.

    Args:
        imgs (List[torch.Tensor] or Image.Image): a list of images with size [224, 224, 3]
    """
    
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            # print(f'Type of input: torch.Tensor')
            img = T.ToPILImage()(img.to('cpu'))
        else:
             print(f'show_images only support showing torch.Tensor, but now is {type(img)}!')
             return
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
    plt.show()
    
def coco_info(cocoDataset: CocoCaptions):
    """Print information of CocoCaptionReshape instance 

    Args:
        cocoDataset (CocoCaptionReshape): cococap dataset with uniform size
    """
    print('-'*40)
    print(cocoDataset.coco.info())
    print('-'*40)
    print('Number of samples:', len(cocoDataset))
    img, cap = cocoDataset[0]  # load first sample
    
    if isinstance(img, torch.Tensor):
        print("First image's size:", img.size())
    elif isinstance(img, Image.Image):
        print("First image's size:", img.size)
    else:
        print(f'coco_info does not support showing {type(img)}!')
        
    if isinstance(cap, list):
        print(f"Target type: {type(cap)}, 'Length: {len(cap)}")
    elif isinstance(cap, torch.Tensor):
        print(f"Target type {type(cap)}:, shape: {cap.size()}")
    else:
        print(f"Target type {type(cap)}")
    print('-'*40)
      
         

if __name__ == '__main__':
    # require coco_val2017 to be download
    image_dir = "data/coco/val2017"
    ann_file = 'data/coco/ann_trainval2017/captions_val2017.json'

    # Construct Dataset
    print('-'*40)
    transform = T.PILToTensor()  # want raw image: None
    coco_val2017 = CocoCaptionReshape(root = image_dir,
                            annFile = ann_file,
                            transform=transform)
    print('-'*40)
    # print dataset info
    # coco_info(coco_val2017)
    
    # Split dataset
    train_ratio = 0.7
    seed = 1
    
    train_data, test_data = rand_split(coco_val2017, train_ratio=train_ratio, seed=seed)
    print(f'Split dataset into {len(train_data)} training samples and {len(test_data)} test samples, train_ratio = {train_ratio}.')
    
    # Build Dataloader for traing and testing
    batch_size = 64
    train_ratio = 0.7
    shuffle = False
    n_workers = 0
    
    train_dataloader = DataLoader(train_data, batch_size, shuffle, num_workers=n_workers)
    test_dataloader = DataLoader(test_data, batch_size, shuffle,num_workers=n_workers)
    
    # print info of dataloader
    print(train_dataloader.batch_size)
    for train_img, tarin_cap in train_dataloader:
        print(f"Image batch shape: {train_img.size()}")
        print(f"Caption batch shape: {len(tarin_cap)}, {type(tarin_cap)}")
        print('-'*40)
        break
    
    # show one sample
    img, cap = coco_val2017[0] 
    imgs = [img]
    print(f"Show one sample:\nCaption\n{cap}")
    show_images(imgs)
    
