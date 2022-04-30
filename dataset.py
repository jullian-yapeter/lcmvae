import os
from PIL import Image
from typing import Any, Callable, Optional, Tuple, List

import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms as T
from torchvision.ops import box_area

from torch.utils.data import DataLoader

import numpy as np
from warnings import warn

from transformers import AutoFeatureExtractor

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
# NOTE: deprecated for segmentation task due to one image may have multiple segmenations 
# TODO: but his class could be useful for object detection and could be modified for one segmentation mask per image.
class MyCocoDetection(CocoDetection):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        warn("MyCocoDetection is deprecated", DeprecationWarning, stacklevel=2)
        # for one image
        id = self.ids[index]
        image = self._load_image(id)
        # List[Dictionary]: target coco_ann file for an image
        coco_anns = self._load_target(id) # self.coco.loadAnns()
        
        # number of objects in the image
        num_objs = len(coco_anns) 
        
        # segments and rectangle boxes for every object in such image
        boxes = []
        segments = []
        segment_ids = []
        # NOTE: The ouput of coco.annToMask() is np.dnarray. creating a tensor from a list of numpy.ndarrays is extremely slow
        # NOTE: Hence, we initalize segment_masks as np.empty((0, *img.shape))
        # [x]: If use seg_mask, the output of annToMask is np.ndarray, tranform it to segment is too slow.
        segment_masks = np.empty(shape=(0, *image.size[1:]))
        segment_areas = []
        category_ids = []
        iscrowd = []
        
        for i in range(num_objs):
            # segmentation (ploy)
            # segments.append(coco_anns[i]['segmentation'])
            # segmentation id
            segment_ids.append(coco_anns[i]['id'])
            # FIXME: segmentation mask with different size, but acturally training don't need it.
            # # segmentation mask
            # seg_mask = self.coco.annToMask(ann=coco_anns[i])
            # segment_masks = np.append(segment_masks, [seg_mask], axis=0)
            # segmentation area
            segment_areas.append(coco_anns[i]['area'])
            
            # Bounding boxes for objects
            # In coco format, bbox = [xmin, ymin, width, height]
            # In PyTorch, the input should be [xmin, ymin, xmax, ymax]
            xmin = coco_anns[i]['bbox'][0]
            ymin = coco_anns[i]['bbox'][1]
            xmax = xmin + coco_anns[i]['bbox'][2]
            ymax = ymin + coco_anns[i]['bbox'][3]
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
            # box_masks.append(self.coco.annToMask(coco_anns[i]))
            
            # other info for one object
            category_ids.append(coco_anns[i]['category_id'])
            iscrowd.append(coco_anns[i]['iscrowd'])
        
        # transform variables to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # no tensor segments for image, cuz the each segment list has different size
        # segments = torch.as_tensor(segments, dtype=torch.float32)
        # segment_masks =  torch.as_tensor(segment_masks, dtype=torch.float32)
        segment_areas = torch.as_tensor(segment_areas, dtype=torch.float32)
        segment_ids = torch.as_tensor(segment_ids, dtype=torch.int32)
        category_ids = torch.as_tensor(category_ids, dtype=torch.int32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int8)
        
        # FIXME: boxes_area will cause errors when running data loader
        # FIXME: IndexError: too many indices for tensor of dimension 1
        # Size for each box
        # boxes_area = box_area(torch.as_tensor(boxes))  
        
        # # Labels (one segment only one class: target class or background)
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        
        # Customized annotation in dictionary format
        segment_ann = {}
        
        segment_ann['image_id'] = id  # one int can't send to GPU
        segment_ann['boxes'] = boxes
        # segment_ann['boxes_area'] = boxes_area
        # segment_ann['segments'] = segments
        # segment_ann['segments_mask'] = segment_masks
        segment_ann['segment_ids'] = segment_ids
        segment_ann['segment_areas'] = segment_areas
        segment_ann['category_ids'] = category_ids
        segment_ann['iscrowd'] = iscrowd
        
        # one ann in coo_anns, which means one segentation
        # ann = {
        #     "id": int, 
        #     "image_id": int, 
        #     "category_id": int, 
        #     "segmentation": RLE or [polygon], 
        #     "area": float, 
        #     "bbox": [x,y,width,height], 
        #     "iscrowd": 0 or 1,
        # }
        
        target = segment_ann
        
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


class MyCocoCaption(CocoDetection):
    """Create a custom Dataset from torchvision.datasets.CocoDetection 
    (source: https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html#CocoDetection)
    
    It requires the COCO API to be installed. COCO API for python: `pycocotools`
    
    _load_image: 
        Images in COCO datasets do not have a uniform shape, so
        reshape each image (3, _, _) -> (3, IMG_W, IMG_H), default IMG_W = IMG_H = 224
        # NOTE: reshaping images will reduce images' quality, does it okay?
        # NOTE: ViT : resize to 256 -> do center crop to 224 -> transform images to tensor and normalize them. All these step could finished automatically by feature_extractor()
        # [x]: use AutoFeatureExtractor() to reshape images
        
    _load_target:
        Each image corresponds to 5 captions.
        CocoDetection return [{'caption': str, 'id': 372891, 'image_id': 139}, ...]
        CocoCaption return a list of captions, List[str]
        Ours return one string: combine 5 sentences for each image -> 1 string
        # [x]: two ways ot handle 5-sentence captions per image (combine or pick one) 
    """
    
    def __init__(
        self,
        from_pretrained: str,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        self.from_pretrained = from_pretrained
        self.feature_extractor =  AutoFeatureExtractor.from_pretrained(self.from_pretrained)
        
    # NOTE: Please choose pick first one caption or combine 5 captions into one. `mode` in ['pick', 'combine']
    mode = 'combine'
    def _captions2str(self, id: int, mode=mode) -> str:
        if mode not in ['pick', 'combine']:
            raise ValueError(f"mode should be 'pick' or combine, but now is {mode}")
        # captions: List[str]
        captions = [ann["caption"] for ann in super()._load_target(id)]
        if mode == 'pick':
            return captions[0]
        return ' '.join(captions)
    
    def _load_image(self, id: int) -> Image.Image:
        # NOTE: constructing AutoFeatureExtractor for each image is pretty slow, leave it outside of _load_image()
        path = self.coco.loadImgs(id)[0]["file_name"]
        raw_img = Image.open(os.path.join(self.root, path)).convert("RGB")
        with torch.no_grad():
            encoding = self.feature_extractor(images=raw_img, return_tensors="pt")
        
        return encoding['pixel_values'][0]
    
    # def _load_target(self, id: int) -> List[str]:
    #     return [ann["caption"] for ann in super()._load_target(id)]

    def _load_target(self, id: int) -> List[str]:
        return self._captions2str(id)


class MyCocoCaptionDetection(MyCocoCaption):
    def __init__(
        self,
        from_pretrained: str,
        root: str,
        annFile: str,
        detAnnFile: str,
        superclasses: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(from_pretrained, root, annFile,
                         transform, target_transform, transforms)
        from pycocotools.coco import COCO
        self.det = COCO(detAnnFile)
        self.cat_ids = self.det.getCatIds(supNms=superclasses)
        img_ids = []
        for cat in self.cat_ids:
            img_ids.extend(self.det.getImgIds(catIds=cat))
        self.img_data = self.det.loadImgs(img_ids)
        self.resizer = T.Resize((IMG_H, IMG_W))

    def __len__(self) -> int:
        return len(self.img_data)
        # return 32

    def _segment_mask(self, id: int) -> torch.LongTensor:
        ann_ids = self.det.getAnnIds(
            imgIds=id,
            catIds=self.cat_ids,
            iscrowd=None
        )
        anns = self.det.loadAnns(ann_ids)
        mask = np.max(np.stack([self.det.annToMask(ann) * ann["category_id"]
                                for ann in anns]), axis=0)
        mask = torch.LongTensor(mask).unsqueeze(0)
        
        return mask

    def _load_target(self, id: int):
        return (super()._captions2str(id), self._segment_mask(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.img_data[index]['id']
        image = self._load_image(id)
        caption, mask = self._load_target(id)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, (caption, self.resizer(mask))


if __name__ == "__main__":
    # coco_val2017
    data_root = './data'
    dataType = 'val2017'
    image_dir = f'{data_root}/coco/val2017/'
    ann_file = f'{data_root}/coco/ann_trainval2017/captions_{dataType}.json'

    # Construct Dataset
    transform = None
    from_pretrained = 'google/vit-base-patch16-224'

    coco_val2017 = MyCocoCaption(root = image_dir,
                                annFile = ann_file,
                                transform = transform,
                                from_pretrained = from_pretrained)

    # Check the info of dataset, you can ignore this part
    print('-'*40)
    coco_val2017.coco.info()
    print('-'*40)
    print(f'The number of samples: {len(coco_val2017)}')
    first_img, first_cap = coco_val2017[0]
    print(f'Image shape: {first_img.size()}')
    
    # Build Dataloader for traing and testing
    batch_size = 512
    shuffle = False
    n_workers = 0
    # WARN: when n_workers > 0, DataLoader will work slowly due to unknow reasons.

    data_loader = DataLoader(coco_val2017, batch_size, shuffle, num_workers=n_workers)

    # Check: print info for each batch
    i = 0
    for imgs, caps in data_loader:
        print(f'batch_{i}')
        print(f"Image batch shape: {imgs.size()}")
        print(f"Caption batch shape: {len(caps)}")
        i += 1