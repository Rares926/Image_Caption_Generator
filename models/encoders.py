import torch.nn as nn
import torchvision.models as models

class EncoderInception(nn.Module):
    # Encoder model that extracts features using inception_v3
    def __init__(self, embed_size) -> None:
        super(EncoderInception, self).__init__()
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.linear = nn.Linear(self.inception.fc.out_features, out_features=embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # features = self.inception(images).logits # for train
        features = self.inception(images)   # for test
        x = self.linear(features)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class EncoderVGG(nn.Module):
    # Encoder model that extracts features using Vgg16
    def __init__(self, embed_size):
        super(EncoderVGG, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg16.classifier[6] = nn.Linear(self.vgg16.classifier[6].in_features, embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.vgg16(images)
        return self.dropout(nn.ReLU(features))
