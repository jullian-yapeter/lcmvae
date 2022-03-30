from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torchvision import transforms as T
from transformers import AutoFeatureExtractor

from utils import rand_split
import os

try:
  import pycocotools
  print("pycocotools has existed, import successfully")
except:
    print("COCO API is required to be installed.")
    print(f"Do you want to install pycocotools right now? [y/n]")
    install = input()
    if install == 'y':
        # WARN: recommand setting env via conda. pip install pycocotools may bring some errors to numpy.
        # https://github.com/scikit-image/scikit-image/issues/5270
        os.system('conda install pycocotools')
        import pycocotools
    else:
        exit()
        

def data_loader():
    # coco_val2017
    data_root = 'data'
    dataType = 'val2017'
    image_dir = f'{data_root}/coco/val2017/'
    ann_file = f'{data_root}/coco/ann_trainval2017/captions_{dataType}.json'

    coco_val2017 = CocoCaptions(root = image_dir,
                                annFile = ann_file,
                                transform = None)

    # # Split dataset
    # train_ratio = 0.7
    # seed = 1

    # train_dataset, test_dataset = rand_split(coco_val2017, train_ratio=train_ratio, seed=seed)
    # print(f'train_ratio = {train_ratio}')
    # print(f'Split dataset into {len(train_dataset)} training samples and {len(test_dataset)} test samples, ')

    images = []
    captions = []
    for img, cap in coco_val2017:
        images.append(img)
        captions.append(cap[0])
        
    from_pretrained = 'google/vit-base-patch16-224'
    feature_extractor = AutoFeatureExtractor.from_pretrained(from_pretrained)
    encoding = feature_extractor(images=images, return_tensors="pt")
    images = encoding['pixel_values']
        
    return images, captions