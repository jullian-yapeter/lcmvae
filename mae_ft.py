import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, ViTMAEForPreTraining

from PIL import Image
import requests
import matplotlib.pyplot as plt

from mae_ft_trainer import MaeFtTrainer
from dataset import MyCocoCaption
from params import PRETRAIN_DATASET_PARAMS as PDP


##################################
############# params #############
##################################
class MaeFtParams:
    pretrained_model = "facebook/vit-mae-base"

    epochs = 1
    learning_rate = 1e-3
    verbose = True  # output rec_loss or not

    # only fine tune MAE decoder
    is_encoder_frozen = True
    is_decoder_frozen = False

    # # only fine tune MAE encoder
    # is_encoder_frozen = False
    # is_decoder_frozen = True

    # # fine tune end to end
    # is_encoder_frozen = False
    # is_decoder_frozen = False
    

##################################
########## mae fine-tune #########
##################################
def sub_dataset(dataset, subset_len=32):
    # create small dataset and dataloader for testing if model is runnable
    subset = torch.utils.data.Subset(dataset, list(range(0, subset_len)))

    return DataLoader(dataset = subset, 
                      batch_size=PDP.batch_size, 
                      shuffle=PDP.shuffle, 
                      num_workers=PDP.num_workers)

def mae_ft():
    # Load coco dataset
    coco_val2017 = MyCocoCaption(root = PDP.image_dir,
                                annFile = PDP.ann_file,
                                from_pretrained = PDP.from_pretrained)

    # Check the info of dataset, you can ignore this part
    print('-'*40)
    coco_val2017.coco.info()
    print('-'*40)
    print(f'The number of samples: {len(coco_val2017)}')
    first_img, first_cap = coco_val2017[0]
    print(f'Image shape: {first_img.size()}')

    # NOTE: modify dataloader for diff training
    ####################################################################
    # # Build Dataloader for pretrain on coco_val2017
    # data_loader = DataLoader(dataset = coco_val2017, 
    #                         batch_size=PDP.batch_size, 
    #                         shuffle=PDP.shuffle, 
    #                         num_workers=PDP.num_workers)
    
    # Build a small dataload for testing if model is runnable
    subset_len = 512
    data_loader_small = sub_dataset(coco_val2017, subset_len=subset_len)
    
    # FIXME: replace `data_loader_small` with real train and val dataloaders
    train_dataloader = data_loader_small
    val_dataloader = data_loader_small
    ####################################################################
    
    # build model and set params 
    pretrained_model = MaeFtParams.pretrained_model
    # MaeFtTrainer will send model to GPU when init
    model = ViTMAEForPreTraining.from_pretrained(pretrained_model) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = MaeFtParams.epochs
    learning_rate = MaeFtParams.learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    # ViTMAEForPreTraining includes loss func: MSE, so no need to build a criterion
    # criterion = nn.MSELoss() 

    verbose = MaeFtParams.verbose  # output rec_loss or not

    # only fine tune MAE decoder
    is_encoder_frozen = MaeFtParams.is_encoder_frozen
    is_decoder_frozen = MaeFtParams.is_decoder_frozen

    mae_ft_trainer = MaeFtTrainer(
        model=model, 
        is_encoder_frozen=is_encoder_frozen, 
        is_decoder_frozen=is_decoder_frozen, 
        train_data_loader=train_dataloader, 
        eval_data_loader=val_dataloader,
        optimizer=optimizer,
        device=device, epochs=epochs, 
        verbose = verbose, is_with_caption = False, args = None
    )
    
    print("=" * 60)
    print("Fine-tune MAE")
    print("-" * 60)
    train_rec_losses = mae_ft_trainer.train()
    
    print("-" * 60)
    print("Evaluate fine-tuned MAE")
    print("-" * 60)
    eval_rec_losses = mae_ft_trainer.evaluate()
    print("=" * 60)   
    
    return model, train_rec_losses, eval_rec_losses


#############################################
##### fit one image into fine-tuned mae #####
#############################################
feature_extractor = AutoFeatureExtractor.from_pretrained(MaeFtParams.pretrained_model)
# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std = np.array([0.229, 0.224, 0.225])
imagenet_mean = torch.tensor(feature_extractor.image_mean)
imagenet_std = torch.tensor(feature_extractor.image_std)

def show_image(img_unpatch, title=''):
    # image is [3, H, W]
    assert img_unpatch.shape[0] == 3
    img_show = torch.einsum('chw->hwc',img_unpatch)
    # Clamps all elements in input into the range [0, 255].
    img_show = torch.clip((img_show * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    plt.imshow(img_show)
    plt.title(title, fontsize=16)
    plt.axis('off')
    
def show_one_image(img_inputs, pixel_mask, pixel_pred):
    # original, patch mask, masked, reconstruction, reconstruction + visible
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    
    plt.figure(figsize=(25, 5))
    plt.subplot(1, 5, 1)
    show_image(img_inputs.pixel_values[0], "original")
    
    plt.subplot(1, 5, 2)
    img_masked = img_inputs.pixel_values * (1 - pixel_mask[0])
    show_image(pixel_mask[0], "patch mask")

    img_masked = img_inputs.pixel_values * (1 - pixel_mask[0])
    plt.subplot(1, 5, 3)
    show_image(img_masked[0], "masked")

    plt.subplot(1, 5, 4)
    show_image(pixel_pred[0], "reconstruction")

    plt.subplot(1, 5, 5)
    im_paste = img_inputs.pixel_values[0] * (1 - pixel_mask) + pixel_pred[0] * pixel_mask
    show_image(im_paste[0], "reconstruction + visible")

    name = "MAE_ft_test_one_image"
    plt.savefig(f"output/{name}.jpg")

def run_one_img(url='', img_path='', model=None):
    assert (url!='' or img_path!=""), "please input an image url or local path"
    assert model != None, "please set `model`"
    
    # load image
    if url != "":
        image = Image.open(requests.get(url, stream=True).raw)
    elif img_path != "":
        image = Image.open(img_path)
    img_inputs = feature_extractor(images=image, return_tensors="pt")
    
    # fit into the model
    img_outputs = model(**img_inputs)
    loss = img_outputs.loss  # tensor(0.3091, grad_fn=<DivBackward0>)
    print(f'rec_loss: {loss}')
    mask = img_outputs.mask
    ids_restore = img_outputs.ids_restore
    # print(img_outputs.logits.size())  # 16*16*3 = 768
    
    mask_ratio = model.config.mask_ratio
    patch_size = model.config.patch_size
    
    # pixel-wise mask
    # patch mask -> unpatch_mask -> pixel mask projection: 
    # torch.Size([1, 196]) -> torch.Size([1, 196, 768]) -> torch.Size([1, 3, 224, 224])
    unpatch_mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 *3)
    pixel_mask = model.unpatchify(unpatch_mask) 
    # print(f'patch mask -> unpatch_mask -> pixel mask projection: \n{mask.size()} -> {unpatch_mask.size()} -> {pixel_mask.size()}')

    # pixel-wise prediction
    # pred of decoder -> pixel-wise pred: 
    # torch.Size([1, 196, 768]) -> torch.Size([1, 3, 224, 224])
    img_outputs.keys()
    pixel_pred = model.unpatchify(img_outputs.logits) 
    # print(f'pred of decoder -> pixel-wise pred: \n{img_outputs.logits.size()} -> {pixel_pred.size()}')
    
    show_one_image(img_inputs, pixel_mask, pixel_pred)

    # return loss, mask, ids_restore, img_inputs, pixel_mask, pixel_pred
 

if __name__=="__main__":
    # fine-tune mae 
    model, train_rec_losses, eval_rec_losses = mae_ft()
    
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(1)
    
    print("Test the fine-tuned model on one image")
    # test the finetuned model on one image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    run_one_img(url=url, model=model)  
    
    
    