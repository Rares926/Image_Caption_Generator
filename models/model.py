import torch
import torch.nn as nn
from models.encoders import EncoderInception, EncoderVGG, EncoderResNet, EncoderViT # noqa
from models.decoders import DecoderRNN


class ImageCaptioningModel(nn.Module):
    def __init__(
            self, embed_size, hidden_size, vocab_size, encoder_name="inception"
            ):
        super(ImageCaptioningModel, self).__init__()

        encoder_name_dict = {
            "inception": EncoderInception(embed_size),
            "vgg": EncoderVGG(embed_size),
            "res_net": EncoderResNet(embed_size),
            "vit": EncoderViT(embed_size),
        }

        self.encoderCNN = encoder_name_dict[encoder_name]
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                # we take the predicted word with the maximum value
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                # if this is eos we will stop because the sentence is done
                if vocabulary.itos[predicted.item()] == "<END>":
                    break

        caption = [vocabulary.itos[idx] for idx in result_caption][1:-1]
        sentence = " ".join(caption)

        return sentence.capitalize()
