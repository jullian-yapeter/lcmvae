import torch
import json
from statistics import mean, variance



def save_checkpoint(model, name=None):
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

def log_losses(total_losses, rec_losses, kl_losses, name):
    loss_log = {}

    loss_log["means"] = {}
    loss_log["means"]["total_loss"] = mean(total_losses)
    loss_log["means"]["rec_loss"] = mean(rec_losses)
    loss_log["means"]["kl_loss"] = mean(kl_losses)

    loss_log["variances"] = {}
    loss_log["variances"]["total_loss"] = variance(total_losses)
    loss_log["variances"]["rec_loss"] = variance(rec_losses)
    loss_log["variances"]["kl_loss"] = variance(kl_losses)

    with open(f"output/{name}.json", "w") as outfile:
        json.dump(loss_log, outfile)

