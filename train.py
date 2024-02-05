import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data.data import get_loader
from models.model import ImageCaptioningModel
from models.utils import load_checkpoint, save_checkpoint
from pathlib import Path
import pickle


train_pipeline = {
    # "train_1": ["vgg", "lstm"],
    "train_2": ["inception", "lstm"],
    "train_3": ["res_net", "lstm"],
    "train_4": ["vit", "lstm"]
}


def train(encoder_decoder_to_use):

    if encoder_decoder_to_use[0] == "vit":
        train_loader, dataset = get_loader(
            Path("datsets/flickr8k"),
            transform=transforms.Compose([transforms.Resize((356, 356)),
                                          transforms.RandomCrop((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]), # noqa
            flag="RGB")
    else:
        train_loader, dataset = get_loader(
            Path("datsets/flickr8k"),
            transform=transforms.Compose([transforms.Resize((356, 356)),
                                          transforms.RandomCrop((299, 299)), # noqa
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]), # noqa
            flag="RGB")

    # - set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_model = False
    save_model = False
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    learning_rate = 3e-4  # as Andrej Karpathy jockingly said this is the best learning rate for Adam # noqa
    num_epochs = 50

    # initialize model, loss etc
    model = ImageCaptioningModel(
        embed_size,
        hidden_size,
        vocab_size,
        encoder_name=encoder_decoder_to_use[0]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(
            "checkpoints/inception/lstm/_model_checkpoint_20.pth.tar",
            model,
            optimizer)

    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])

    # pass the model to train mode
    model.train()

    # Only finetune the CNN
    for name, param in model.encoderCNN.feature_extractor.named_parameters():
        param.requires_grad = train_CNN

    # epochs_to_save = [10, 20, 40, 60, 100]

    epochs_to_save = [5, 10, 15, 20, 30, 40, 50]  # finetune only

    losses = []

    for _epoch in range(num_epochs):

        print("Started epoch {}".format(_epoch+1))
        for _idx, (imgs, captions) in tqdm(enumerate(train_loader),
                                           total=len(train_loader),
                                           leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # we send the captions without the last one so
            # that the model learn to predict it
            outputs = model(imgs, captions[:-1])
            loss = loss_fn(outputs.reshape(-1, outputs.shape[2]),
                           captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        print("Epoch loss -> {}".format(loss))
        losses.append(loss)

        if save_model:
            if _epoch+1 in epochs_to_save:
                checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
                save_checkpoint(checkpoint,
                                filename="checkpoints/{}/{}/model_checkpoint_{}.pth.tar".format( # noqa
                                    encoder_decoder_to_use[0],
                                    encoder_decoder_to_use[1],
                                    str(_epoch+1))) # noqa

    # ! check if this works
    with open("checkpoints/{}/{}/losses_{}.pkl".format( # noqa
                                    encoder_decoder_to_use[0],
                                    encoder_decoder_to_use[1],
                                    str(_epoch+1)), "wb") as file:
        pickle.dump(losses, file)


if __name__ == "__main__":
    for key, value in train_pipeline.items():
        print("#"*6 + "Started {}".format(key) + "#"*6)
        train(value)
