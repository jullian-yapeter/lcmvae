# lcmvae
Language-Conditioned Masked Variational Autoencoder

## dev_lei

**Data processing**

- [x] download_coco_val2017.sh 
  
  Use the following command line to download `2017 Val images [5K/1GB]` and  `2017 Train/Val annotations [241MB]` into `data/coco/`. 

  Dataset `coco_val2017` isn't large, which could be download within 10 min.
  
  ```
  sh download_coco_val2017.sh 
  ```
  
- [x] Create a custom Dataset class `CocoCaptionReshape` and DataLoader
    `CocoCaptionReshape` reshape all images' size to (3, 224, 224)

- [ ] Create dataset for downstream test: segmentation
    
- [ ] [Demo: pipeline of data processing](./dataload_demo.ipynb) 

    - [ ] Example on how to load data for lcmvae 

        FeatureExtractor focus on load data on hugging face hub, we will customize dataload with PyTorch's `Dataset` and `DataLoader`

    - [ ] simple train loop for lcmvae

    - [ ] Demo like [MAE's](https://github.com/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb) showing original, masked ,reconstruction, reconstruction+visible, image

- [x] Masking: finished with `ViTMAE`

- [ ] Creating a Sufficiently Large Dataset




Insteresting! https://arxiv.org/pdf/1907.01710.pdf

Question:
- Could reshaping shape of images be a kind of data augmentation?