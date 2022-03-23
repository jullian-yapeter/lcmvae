import os
from PIL import Image
from typing import List

import torch
from torchvision.datasets import CocoDetection
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

class CocoCaptionReshape(CocoDetection):
    """Create a custom Dataset from torchvision.datasets.CocoDetection 
    (source: https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html#CocoDetection)
    
    It requires the COCO API to be installed. COCO API for python: `pycocotools`
    
    _load_image: 
        Images in COCO datasets do not have a uniform shape, so
        reshape each image (3, _, _) -> (3, IMG_W, IMG_H), default IMG_W = IMG_H = 224
        TODO: reshaping images will reduce images' quality, does it okay?
    
    _load_target:
        Each image corresponds to 5 captions.
        CocoDetection return [{'caption': str, 'id': 372891, 'image_id': 139}, ...]
        CocoCaption return a list of captions, List[str]
        Ours return one string: combine 5 sentences for each image -> 1 string
        TODO: decise how to use 5-sentence captions for each image (combine or pick one) 
    """
    
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB").resize((IMG_W, IMG_H))

    def _load_target(self, id: int) -> List[str]:
        combined_cap = ''
        for ann in super()._load_target(id):
            combined_cap += ann["caption"] + ' '
        return combined_cap


def show_images(imgs: List[torch.Tensor]):
    """Transform image Tensors to PIL.Image and then show them.

    Args:
        imgs (List[torch.Tensor]): a list of images with size [224, 224, 3]
    """
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
    plt.show()
    
def coco_info(cocoDataset: CocoCaptionReshape):
    """Print information of CocoCaptionReshape instance 

    Args:
        cocoDataset (CocoCaptionReshape): cococap dataset with uniform size
    """
    print('-'*40)
    print(cocoDataset.coco.info())
    print('-'*40)
    print('Number of samples:', len(cocoDataset))
    img, cap = cocoDataset[0]  # load first sample
    assert tuple(img.shape) == (3, 224, 224)
    print("Image Size:", img.size())
    print("One Caption:", type(cap), 'length', len(cap))
    print('-'*40)
      
def rand_split(dataset, train_ratio=0.7, seed=None):
    """Split dataset by train_ratio

    Args:
        dataset (torch.utils.data.Dataset): whole dataset 
        train_ratio (float, optional): the ratio of traing samples. Defaults to 0.7.
        seed (None or int, optional): random seed. Defaults to None.

    Returns:
        train_data: torch.utils.data.Dataset
        test_data: torch.utils.data.Dataset
    """
    # set random seed to control randomness
    if seed:
        torch.manual_seed(seed)

    train_data, test_data = random_split(dataset, [round(train_ratio * len(dataset)), len(dataset) - round(train_ratio * len(dataset))])
    assert len(dataset) == (len(train_data) + len(test_data))
    
    return train_data, test_data
    
    

if __name__ == '__main__':
    # require coco_val2017 to be download
    image_dir = "data/coco/val2017"
    ann_file = 'data/coco/ann_trainval2017/captions_val2017.json'

    # Construct Dataset
    print('-'*40)
    coco_val2017 = CocoCaptionReshape(root = image_dir,
                            annFile = ann_file,
                            transform=T.PILToTensor())
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
    
