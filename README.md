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

- [x] [Demo: pipeline of data processin](./dataload_demo.ipynb)

- [ ] masking
- [ ] Creating a Sufficiently Large Dataset