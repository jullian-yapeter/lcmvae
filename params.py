class PRETRAIN_PARAMS:
    epochs = 2
    learning_rate = 1e-4
    beta = 0

class TRAIN_PARAMS:
    epochs = 2
    learning_rate = 1e-4
    beta = 0

class PRETEST_PARAMS:
    beta = 0

class TEST_PARAMS:
    beta = 0
    
class PRETRAIN_DATASET_PARAMS:
    data_root = './data'
    dataType = 'val2017'  # dataType: 'train2017' or 'val2017'
    image_dir = f'{data_root}/coco/{dataType}/'
    det_ann_file = f'{data_root}/coco/ann_trainval2017/instances_{dataType}.json'
    ann_file = f'{data_root}/coco/ann_trainval2017/captions_{dataType}.json'
    transform = None
<<<<<<< HEAD
    # [ ]: which from_pretrained?
=======
    # NOTE: set proper from_pretrained for dataset
    # VitEncoder: "google/vit-base-patch16-224-in21k"
    # VitEncoder: 'facebook/vit-mae-base'
>>>>>>> 4d32d50a6c7adde26b2e57d2050796c8d56a5f67
    from_pretrained = 'facebook/vit-mae-base'
    
    # DataLoader
    batch_size = 64
    shuffle = False
    num_workers = 0
    # WARN: when n_workers > 0, DataLoader will work slowly due to unknow reasons.
     
class MODE_PARAMS:
    experiment_name = "no_mask"
    pretrain = True
    pretest = False
    train = False
    test = False
    
