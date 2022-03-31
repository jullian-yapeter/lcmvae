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
    image_dir = f'{data_root}/coco/val2017/'
    ann_file = f'{data_root}/coco/ann_trainval2017/captions_{dataType}.json'
    transform = None
    from_pretrained = 'google/vit-base-patch16-224'
    
    # DataLoader
    batch_size = 1
    shuffle = False
    num_workers = 0
    # WARN: when n_workers > 0, DataLoader will work slowly due to unknow reasons.
    
    
