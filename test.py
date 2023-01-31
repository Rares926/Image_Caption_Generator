import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data.data import get_loader
from models.model import ImageCaptioningModel
from models.utils import load_checkpoint
from pathlib import Path
from PIL import Image


def test():

    _, dataset = get_loader(
        Path("Z:/Master I/NLP - Foundations NLP/Image_Caption_Generator/datasets/flickr8k"),
        flag="RGB"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1

    # initialize model, loss etc
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size).to(device)

    load_checkpoint("Z:/Master I/NLP - Foundations NLP/Image_Caption_Generator/checkpoints/inception/model_checkpoint.pth.tar",
                    model)

    img_transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img = img_transform(Image.open("datasets/test_data/girls.jpg").convert("RGB")).unsqueeze(0)
    print(model.caption_image(test_img.to(device), dataset.vocab)[1:-1])

if __name__ == "__main__":
    test()