import imp
import torch
import json
from statistics import mean, variance
from torch.utils.data import random_split
import os
import http.client as httplib

from prettytable import PrettyTable


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
    if name is not None:
        torch.save(model, f"{save_dir}/{checkpoint_file}_{name}")
    else:
        torch.save(model, f"{save_dir}/{checkpoint_file}")


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
    conn = httplib.HTTPSConnection("8.8.8.8", timeout=5)
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
