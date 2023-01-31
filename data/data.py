import pandas as pd  
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  
from data.utils import Vocabulary, MyCollate
import torchvision.transforms as transforms
from pathlib import Path


class ImageCaptioningDataset(Dataset):
    def __init__(self,
                 dataset_dir: Path,
                 transform=None,
                 freq_threshold: int=5,
                 flag: str="RGB"):

        self.root_dir = dataset_dir/"images"
        self.df = pd.read_csv(dataset_dir/"captions.txt")
        self.transform = transform
        self.flag = flag

        # Get img, caption columns
        self.imgs,self.captions = self.df["image"],self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(self.root_dir/str(img_id)).convert(self.flag)

        if self.transform is not None:
            img = self.transform(img)

        num_caption = [self.vocab.stoi["<START>"]]
        num_caption += self.vocab.to_numerical(caption)
        num_caption.append(self.vocab.stoi["<END>"])

        return img, torch.tensor(num_caption)


def get_loader(root_folder,
              transform="None",
              flag="L",
              batch_size=32,
              shuffle=True):
    """
    Method for generating thed Dataset and Dataloader objects needed.

    Args:
        root_folder (Path): Path to the root folder of the dataset
        transform ("None" or torchvision.transforms): Transforms to be applied on the images.
        flag (str, optional): How to load the images. Defaults to L.
        batch_size (int, optional):Defaults to 32.
        shuffle (bool, optional): Wheter to shuffle or nor the data. Defaults to True.

    Returns:
        (torch.utils.data.DataLoader, torch.utils.data.Dataset): Returns the dataset and the dataloader to be used for train.
    """
    dataset = ImageCaptioningDataset(root_folder,
                                     transform = transform,
                                     flag = flag)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=MyCollate(pad_idx=dataset.vocab.stoi["<PAD>"]),
    )

    return loader, dataset


