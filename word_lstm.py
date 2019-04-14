import pickle
import configparser

from torch import nn
import torch
import torchtext
import numpy as np
import time

config = configparser.ConfigParser()
config.read("config.ini")

config = config["WORD-LSTM"]

seq_length = int(config["seq_length"])
batch_size = int(config["batch_size"])
input_size = int(config["input_size"])
hidden_size = int(config["hidden_size"])
num_layers = int(config["num_layers"])
num_epochs = int(config["num_epochs"])
learning_rate = float(config["learning_rate"])
dropout = float(config["dropout"])


class LSTMWordNet(nn.Module):
    def __init__(self, in_size, hidden_size, nb_layer, nb_classes, device, dropout):
        super(LSTMWordNet, self).__init__()
        self.hidden_size = hidden_size
        self.nb_layer = nb_layer
        self.nb_classes = nb_classes
        self.emb = torchtext.vocab.GloVe()
        self.lstm = nn.LSTM(
            in_size, hidden_size, nb_layer, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, nb_classes)
        self.device = device

    def forward(self, x, batch_size=config["batch_size"]):
        x = self.emb(x).view(-1, config["seq_length"], config["input_size"])
        h0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    fics = []

    # Load Fanfics, 49999 in total
    with open("./fics.pkl", "rb") as file:
        fics = pickle.load(file)
        fics = [fics[4]]  # begin with only this much

