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
    # MyCocoCaption
    # coco_val2017
    data_root = './data'
    dataType = 'val2017'
    image_dir = f'{data_root}/coco/{dataType}/'
    det_ann_file = f'{data_root}/coco/ann_trainval2017/instances_{dataType}.json'
    ann_file = f'{data_root}/coco/ann_trainval2017/captions_{dataType}.json'
    transform = None
    # [ ]: which from_pretrained?
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
    
