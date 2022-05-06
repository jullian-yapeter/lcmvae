import torch
import json
from statistics import mean, variance
from torch.utils.data import random_split
import os
import http.client as httplib

from prettytable import PrettyTable

from transformers import AutoFeatureExtractor, ViTMAEForPreTraining
from PIL import Image
import requests
import matplotlib.pyplot as plt

from masks import PixelMask, PatchMask


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def save_checkpoint(model, checkpoint_file=None, name=None, save_dir="saved_models"):
    # automatically create saved_models folder
    try:
        os.mkdir('saved_models')
        print("Directory " , 'saved_models' ,  " Created ") 
    except FileExistsError:
        pass
    
    checkpoint_file = checkpoint_file if checkpoint_file is not None else model.checkpoint_file
    if name is not None:
        torch.save(model.state_dict(), f"{save_dir}/{checkpoint_file}_{name}")
    else:
        torch.save(model.state_dict(), f"{save_dir}/{checkpoint_file}")


def save_model(model, checkpoint_file=None, name=None, save_dir="saved_models"):
    # automatically create saved_models folder
    try:
        os.mkdir('saved_models')
        print("Directory ", 'saved_models',  " Created ")
    except FileExistsError:
        pass

    checkpoint_file = checkpoint_file if checkpoint_file is not None else model.checkpoint_file
    save_path = f"{save_dir}/{checkpoint_file}_{name}" \
        if name else f"{save_dir}/{checkpoint_file}"
    torch.save(model, save_path)
    print(f"Checkpoint Saved: {save_path}")


def load_checkpoint(model, checkpoint_file=None, name=None, save_dir=None):
    if checkpoint_file:
        model.load_state_dict(
            torch.load(checkpoint_file, map_location=model.device))
        print(f"loaded {checkpoint_file}")
    else:
        save_dir = save_dir if save_dir is not None else 'saved_models'
        if name is not None:
            model.load_state_dict(
                torch.load(f"{save_dir}/{model.checkpoint_file}_{name}", map_location=model.device))
            print(f"loaded {save_dir}/{model.checkpoint_file}_{name}")
        else:
            model.load_state_dict(
                torch.load(f"{save_dir}/{model.checkpoint_file}", map_location=model.device))
            print(f"loaded {save_dir}/{model.checkpoint_file}")

def log_losses(losses, name):
    loss_log = {}

    loss_log["means"] = {}
    for loss_name, loss_list in losses.items():
        loss_log["means"][loss_name] = mean(loss_list)


    loss_log["variances"] = {}
    for loss_name, loss_list in losses.items():
        if len(loss_list) > 1:
            loss_log["variances"][loss_name] = variance(loss_list)

    with open(f"output/{name}.json", "w") as outfile:
        json.dump(loss_log, outfile)


def has_internet():
    conn = httplib.HTTPSConnection("8.8.8.8", timeout=2)
    try:
        conn.request("HEAD", "/")
        return True
    except Exception:
        return False
    finally:
        conn.close()


###################
#     Dataset     #
###################
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

    train_dataset, test_dataset = random_split(dataset, [round(train_ratio * len(dataset)), len(dataset) - round(train_ratio * len(dataset))])
    assert len(dataset) == (len(train_dataset) + len(test_dataset))
    
    return train_dataset, test_dataset

def denormalize_torch_to_cv2(im, mean, std):
    im = im.permute(1, 2, 0) * std + mean
    return torch.clip(im * 255, 0, 255).int().cpu().detach().numpy()[:, :, ::-1]


########################
#    Visualization     #
########################
# retrieve mean and std values of images for pretrained MAE
pretrained_model = 'facebook/vit-mae-base'
feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model)
# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std = np.array([0.229, 0.224, 0.225])
imagenet_mean = torch.tensor(feature_extractor.image_mean)
imagenet_std = torch.tensor(feature_extractor.image_std)

def show_image(img_unpatch, title=''):
    # image shape is [3, H, W]
    assert img_unpatch.shape[0] == 3
    img_show = torch.einsum('chw->hwc',img_unpatch)
    img_show = torch.clip((img_show * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    plt.imshow(img_show)
    plt.title(title, fontsize=16)
    plt.axis('off')
    
def mae_show_one_image(img_inputs, pixel_mask, pixel_pred):
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    
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

    plt.show()
    
    
def mae_run_one_img(url='', img_path='', model=None, verbose=True):
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
    loss = img_outputs.loss  # a tensor float with gradient
    if verbose:
        print("rec_loss:", loss)
    mask = img_outputs.mask
    ids_restore = img_outputs.ids_restore
    mask_ratio = model.config.mask_ratio
    patch_size = model.config.patch_size
    
    # create pixel mask based on patch mask
    # patch mask -> unpatch_mask -> pixel mask: 
    # torch.Size([1, 196]) -> torch.Size([1, 196, 768]) -> torch.Size([1, 3, 224, 224])
    unpatch_mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 *3)
    pixel_mask = model.unpatchify(unpatch_mask) 
    if verbose:
        print(f'patch mask -> unpatch_mask -> pixel mask projection: \n{mask.size()} -> {unpatch_mask.size()} -> {pixel_mask.size()}')

    # create pixel prediction (reconstruction) using unpatchifying
    # pred of decoder -> pixel-wise pred: 
    # torch.Size([1, 196, 768]) -> torch.Size([1, 3, 224, 224])
    pixel_pred = model.unpatchify(img_outputs.logits) 
    if verbose:
        print(f'pred of decoder -> pixel-wise pred: \n{img_outputs.logits.size()} -> {pixel_pred.size()}')
    
    mae_show_one_image(img_inputs, pixel_mask, pixel_pred)



def vae_show_one_image(url='', img_path='', model=None, mask_ratio=0.25, 
                       patch_size=16, is_patch=True, verbose=True):
    
    assert (url!='' or img_path!=""), "please input an image url or local path"
    # FIXME: figure out which VAE will be used and the its ouput
    # assert model != None, "please send a VAE `model`"
    
    # load image
    if url != "":
        image = Image.open(requests.get(url, stream=True).raw)
    elif img_path != "":
        image = Image.open(img_path)
    img_inputs = feature_extractor(images=image, return_tensors="pt")
    
    # FIXME: put the image into VAE for reconstruction
    # img_outputs = model(img_inputs[0])
    # reconstruction = img_outputs.reconstruction
    reconstruction = torch.rand(1, 3, 224, 224)  
    # FIXME: calculate loss here
    # loss = 
    # if verbose:
    #     print("rec_loss:", loss)
    
    if is_patch:
        mask_maker = PatchMask(mask_ratio=mask_ratio, patch_size=patch_size)
        masked_image, mask = mask_maker(img_inputs.pixel_values)
    else: # pixel-wise mask
        mask_maker =  PixelMask(mask_ratio=mask_ratio)
        masked_image, mask = mask_maker(img_inputs.pixel_values)
       
    
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    
    plt.subplot(1, 3, 1) 
    # NOTE: the image will be masked in-place, so load it again to show the original image
    show_image(feature_extractor(images=image, return_tensors="pt").pixel_values[0], "Original") 
    
    plt.subplot(1, 3, 2)  
    show_image(masked_image[0], f"Masked Image (mask ratio = {mask_ratio})")

    plt.subplot(1, 3, 3) 
    show_image(reconstruction[0], "Reconstruction")
    
    plt.show()
    
    
mae_with_decoder = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
def show_masked_image(target, mask=None, patch_size=16):
    assert mask != None, "please set `mask`"
    
    unpatch_mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 *3)
    pixel_mask = mae_with_decoder.unpatchify(unpatch_mask)  # [1, 3, 224, 224]
    pixel_mask = torch.einsum('nchw->nhwc', pixel_mask) # [1, 224, 224, 3]
    pixel_mask = pixel_mask.cpu().detach().numpy()  # convert tensor to numpy 
    masked_image = target * (1 - pixel_mask[0]) # (224, 224, 3)
       
    return masked_image
