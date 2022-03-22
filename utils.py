import torch 


def save_checkpoint(model):
    torch.save(model.state_dict(), f"saved_models/{model.checkpoint_file}")

def load_checkpoint(model, checkpoint_file=None):
    if checkpoint_file:
        model.load_state_dict(
            torch.load(checkpoint_file, map_location=model.device))
    else:
        model.load_state_dict(
            torch.load(f"saved_models/{model.checkpoint_file}", map_location=model.device))
