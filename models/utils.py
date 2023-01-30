import torch

def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    """
    Save checkpoint containing state dict for model and optimizer

    Args:
        state (Dict): Dict containing state dict of model and optimizer.
        filename (str, optional): _description_. Defaults to "my_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])