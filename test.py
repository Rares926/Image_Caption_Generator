import torch
import torchvision.transforms as transforms
from data.data import get_loader
from models.model import ImageCaptioningModel
from models.utils import load_checkpoint
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def test():

    _, dataset = get_loader(
        Path("datsets/flickr8k"),
        flag="RGB"
    )

    print(dataset[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(dataset.vocab)

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)

    # initialize model, loss etc
    model = ImageCaptioningModel(
        embed_size, hidden_size, vocab_size, encoder_name="inception"
        ).to(device)

    load_checkpoint("checkpoints/inception/lstm/model_checkpoint_20.pth.tar",
                    model)

    test_transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()

    folder_path = Path("datsets/test_data")
    img_paths = [p for p in folder_path.glob("*") if p.is_file()]

    for img in img_paths:

        loaded_img = Image.open(img).convert("RGB")
        transformed_img = test_transform(loaded_img).unsqueeze(0)

        plt.imshow(loaded_img)
        plt.axis("off")
        plt.title(
            model.caption_image(
                transformed_img.to(device),
                dataset.vocab
                )
            )
        plt.show()


if __name__ == "__main__":
    test()
