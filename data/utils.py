import torch
from torch.nn.utils.rnn import pad_sequence
import nltk


class Vocabulary:
    def __init__(self, freq_threshold: int):

        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, sentence_list):
        """
        Adds words that pass the freq_threshold to the itos and stoi dicts.

        Args:
            sentence_list (list): List of captions.
        """
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in nltk.word_tokenize(sentence.lower()):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def to_numerical(self, text):
        """
        Replaces the words with their appropriate index using the stoi dict.
        Args:
            text (List[str]): List of words
        Returns:
            List[int]: Returns the coverted list.
        """
        tokenized_text = nltk.word_tokenize(text.lower())
        numerical_converted_list = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numerical_converted_list.append(self.stoi[token])
            else:
                numerical_converted_list.append(self.stoi["<UNK>"])
        return numerical_converted_list


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(
            targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
