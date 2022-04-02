import imp
import torch
import json
from statistics import mean, variance
from torch.utils.data import random_split
import os


def save_checkpoint(model, name=None):
    # automatically create saved_models folder
    try:
        os.mkdir('saved_models')
        print("Directory " , 'saved_models' ,  " Created ") 
    except FileExistsError:
        pass
    
    if name is not None:
        torch.save(model.state_dict(), f"saved_models/{model.checkpoint_file}_{name}")
    else:
        torch.save(model.state_dict(), f"saved_models/{model.checkpoint_file}")

def load_checkpoint(model, checkpoint_file=None, name=None):
    if checkpoint_file:
        model.load_state_dict(
            torch.load(checkpoint_file, map_location=model.device))
        print(f"loaded {checkpoint_file}")
    else:
        if name is not None:
            model.load_state_dict(
                torch.load(f"saved_models/{model.checkpoint_file}_{name}", map_location=model.device))
            print(f"loaded saved_models/{model.checkpoint_file}_{name}")
        else:
            model.load_state_dict(
                torch.load(f"saved_models/{model.checkpoint_file}", map_location=model.device))
            print(f"loaded saved_models/{model.checkpoint_file}")

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
    return torch.clip(im * 255, 0, 255).int().detach().numpy()
