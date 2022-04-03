from utils import has_internet
class PRETRAIN_PARAMS:
    epochs = 15
    learning_rate = 1e-4
    beta = 0

class TRAIN_PARAMS:
    epochs = 15
    learning_rate = 1e-4
    beta = 0

class PRETEST_PARAMS:
    beta = 0


class TEST_PARAMS:
    beta = 0
    
class PRETRAIN_DATASET_PARAMS:
    data_root = './data'
    dataType = 'train2017'  # dataType: 'train2017' or 'val2017'
    image_dir = f'{data_root}/coco/{dataType}/'
    det_ann_file = f'{data_root}/coco/ann_trainval2017/instances_{dataType}.json'
    ann_file = f'{data_root}/coco/ann_trainval2017/captions_{dataType}.json'
    transform = None
    # NOTE: set proper from_pretrained for dataset
    # VitEncoder: "google/vit-base-patch16-224-in21k"
    # VitEncoder: 'facebook/vit-mae-base'
    from_pretrained = 'facebook/vit-mae-base' \
        if has_internet() else './saved_models/ViTMAE'
    
    # DataLoader
    batch_size = 128
    shuffle = True
    num_workers = 0
    # WARN: when n_workers > 0, DataLoader will work slowly due to unknow reasons.
    
    
