import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data.data import get_loader
from models.model import ImageCaptioningModel
from models.utils import load_checkpoint, save_checkpoint
from pathlib import Path


def train():

    train_loader, dataset = get_loader(
        Path("Z:/Master I/NLP - Foundations NLP/Image_Caption_Generator/datasets/flickr8k"),
        transform=transforms.Compose([transforms.Resize((356, 356)),
                                      transforms.RandomCrop((299, 299)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
        flag="RGB")

    # - set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_model = False
    save_model = False
    # train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    learning_rate = 3e-4  # as Andrej Karpathy jockingly said this is the best learning rate for Adam
    num_epochs = 100

    # initialize model, loss etc
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint("Z:/Master I/NLP - Foundations NLP/Image_Caption_Generator/checkpoints/inception/model_checkpoint.pth.tar",
                        model,
                        optimizer)

    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])

    # pass the model to train mode
    model.train()

    for param in model.encoderCNN.inception.parameters():
        param.requires_grad = False

    for _epoch in range(num_epochs):

        for _idx, (imgs, captions) in tqdm(enumerate(train_loader),
                                           total=len(train_loader),
                                           leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # we send the captions without the last one so that the model learn to predict it
            outputs = model(imgs, captions[:-1])
            loss = loss_fn(outputs.reshape(-1, outputs.shape[2]),
                           captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        if save_model:
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint)

if __name__ == "__main__":
    train()
