import torch.nn as nn
import torchvision.models as models


class EncoderInception(nn.Module):
    # Encoder model that extracts features using inception_v3
    def __init__(self, embed_size) -> None:
        super(EncoderInception, self).__init__()
        self.feature_extractor = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
            )
        self.linear = nn.Linear(
            self.feature_extractor.fc.out_features, out_features=embed_size
            )

        # self.feature_extractor.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.feature_extractor(images).logits  # for train
        # features = self.feature_extractor(images)   # for test
        x = self.linear(features)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class EncoderVGG(nn.Module):
    # Encoder model that extracts features using Vgg16
    def __init__(self, embed_size):
        super(EncoderVGG, self).__init__()
        self.feature_extractor = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
            )

        self.linear = nn.Linear(
          self.feature_extractor.classifier[6].in_features, embed_size)

        self.feature_extractor.classifier[6] = nn.Identity()

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.feature_extractor(images)
        x = self.linear(features)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class EncoderResNet(nn.Module):
    def __init__(self, embed_size) -> None:
        super(EncoderResNet, self).__init__()
        self.feature_extractor = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
            )
        self.linear = nn.Linear(
            self.feature_extractor.fc.out_features, out_features=embed_size
            )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.feature_extractor(images)
        x = self.linear(features)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class EncoderViT(nn.Module):
    def __init__(self, embed_size) -> None:
        super(EncoderViT, self).__init__()

        # Load a pre-trained Vision Transformer model
        # Adjust the model name as per availability: 'vit_b_16', 'vit_b_32',.
        self.feature_extractor = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Add a custom linear layer to map to the desired embedding size
        self.linear = nn.Linear(
            in_features=self.feature_extractor.heads.head.in_features,  # Use the correct attribute for the output features of your ViT model # noqa
            out_features=embed_size
        )

        # Remove the classification head
        self.feature_extractor.heads = nn.Identity()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # Extract features with the ViT model
        features = self.feature_extractor(images)

        # Apply the custom linear layer, ReLU, and Dropout
        x = self.linear(features)
        x = self.relu(x)
        x = self.dropout(x)

        return x

# class EncoderVGG(nn.Module):
#     # Encoder model that extracts features using Vgg16
#     def __init__(self, embed_size):
#         super(EncoderVGG, self).__init__()
#         self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#         self.vgg16.classifier[6] = nn.Linear(
#           self.vgg16.classifier[6].in_features, embed_size)
#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()

#     def forward(self, images):
#         features = self.vgg16(images)
#         return self.dropout(self.relu(features))
