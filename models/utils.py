import torch

def save_checkpoint(state_dict, filename="model_checkpoint.pth.tar"):
    """
    Save checkpoint containing state dict for model and optimizer

    Args:
        state_dict (Dict): Dict containing state dict of model and optimizer.
        filename (str, optional): _description_. Defaults to "my_checkpoint.pth.tar".
    """
    torch.save(state_dict, filename)
    print(10 * "=" + ">Saved checkpoint")

def load_checkpoint(checkpoint_path: str, model, optimizer="None"):
    """
    Loads the saved state dicts of the model and of the optimizer if necessary

    Args:
        checkpoint_path (str): Path to the saved checkpoint.
        model (torch.nn.Module): Instantiated model.
        optimizer (str or torch.optim, optional): Instantiated optimizer. Defaults to "None".
    """
    print(10 * "=" + "Loading checkpoint")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer != "None":
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(10 * "=" + "Checkpoint loaded model updated")
