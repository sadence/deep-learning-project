
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
hidden_size = int(config["hidden_size"])
num_layers = int(config["num_layers"])
num_epochs = int(config["num_epochs"])
learning_rate = float(config["learning_rate"])
dropout = float(config["dropout"])



class LSTMWordNet(nn.Module):
    def __init__(self, hidden_size, nb_layer, device, dropout):
        super(LSTMWordNet, self).__init__()
        self.hidden_size = hidden_size
        self.nb_layer = nb_layer
        self.glove = torchtext.vocab.GloVe(name='6B') # defqult dim is 300
        self.nb_classes = len(self.glove.itos)
        self.emb = nn.Embedding.from_pretrained(self.glove.vectors, freeze=True, sparse=False)
        self.lstm_input = len(self.glove.vectors[0])
        self.lstm = nn.LSTM(
            self.lstm_input, hidden_size, nb_layer, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, self.nb_classes)
        self.device = device

    def forward(self, x, batch_size=config["batch_size"]):
        x = self.emb(x).view(-1, seq_length, self.lstm_input)
        h0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":

    def to_one_hot(i, total_classes):
        return np.eye(total_classes)[i]

    fics = []

    # Load Fanfics, 49999 in total
    with open("./fics-processed.pkl", "rb") as file:
        fics = pickle.load(file)
        fics = fics[:3]  # begin with only this much

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device being used is {device}')

    total_loss = []
    bleu_scores = []

    model = LSTMWordNet(hidden_size, num_layers, device, dropout).to(device)


    # Prepare Training Data
    dataX = []
    dataY = []

    errors = 0

    # TODO: instead of raw body, sanitize the data
    for j, fic in enumerate(fics):
        fic_arr = fic.body.split()
        for word in fic_arr:
            try:
                model.glove.stoi[word]
            except KeyError:
                print(word)
