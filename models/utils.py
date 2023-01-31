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


def load_checkpoint(checkpoint_path: str, model, optimizer="None"):
    """_summary_

    Args:
        checkpoint (_type_): _description_
        model (_type_): _description_
        optimizer (_type_): _description_
    """
    print(10 * "=" + "Loading checkpoint")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer != "None":
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(10 * "=" + "Checkpoint loaded model updated")
