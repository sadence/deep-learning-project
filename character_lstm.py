"""This script generates a Character LSTM model from the first fanfic in fanfic.csv"""

import pickle
import configparser
import time

from torch import nn
import torch
import torch.utils.data
import numpy as np
from predict import predict_bleu, BLEU_WEIGHTS, mean_bleu
from parse_fics import Fanfic

def num(string):
    """utility function for parsing a string to a number"""
    try:
        return int(string)
    except ValueError:
        return float(string)

def to_one_hot(i, total_classes):
    """utility function for getting a one hot vector from an ordinal"""
    return np.eye(total_classes)[i]

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')
CONFIG = dict(CONFIG['DEFAULT'])

for key in CONFIG:
    CONFIG[key] = num(CONFIG[key])

FICS_PATH = "./fics.csv"

class LSTMNet(torch.nn.Module):
    """LSTMNet defines a character LSTM Network with options"""

    def __init__(self, in_size, hidden_size, nb_layer, nb_classes, seq_length, device, dropout):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.nb_layer = nb_layer
        self.nb_classes = nb_classes
        self.in_size = in_size
        self.seq_length = seq_length
        self.emb = torch.nn.Embedding(nb_classes, in_size)
        self.lstm = torch.nn.LSTM(
            in_size, hidden_size, nb_layer, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, nb_classes)
        self.device = device

    def forward(self, x):
        # print(f'x: {x.shape}')
        # -1 corresponds to batch size (but is smaller for last batch)
        x = self.emb(x).view(-1, self.seq_length, self.in_size)
        # initial states
        # print(f'x: {x.shape}')
        h_0 = torch.zeros(self.nb_layer, x.size(
            0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.nb_layer, x.size(
            0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    fics = []
    chars = set()
    char_to_int = {}

    # Load Fanfics, 49999 in total
    with open("./fics.pkl", 'rb') as file:
        fics = pickle.load(file)
        fics = fics[:1]  # begin with only this much

    # Prepare Vocabulary
    for fic in fics:
        chars.update(set(fic.body))

    chars = sorted(list(chars))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    n_vocab = len(chars)
    n_chars = 0

    print(f"Total Vocabulary: {n_vocab} words: \n{chars}")

    # Prepare Training Data
    dataX = []
    dataY = []

    seq_length = CONFIG['seq_length']

    # TODO: instead of raw body, sanitize the data
    for j, fic in enumerate(fics):
        print(f"Building samples {j}, {len(fic.body)}")
        n_chars += len(fic.body)
        for i in range(0, len(fic.body) - seq_length, 1):
            seq_in = fic.body[i: i + seq_length]
            seq_out = fic.body[i + seq_length]
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])

    n_patters = len(dataX)
    print(f"Total patterns: {n_patters}")

    y = to_one_hot(dataY, n_vocab)

    dataX = np.array(dataX)
    dataX = torch.as_tensor(dataX, dtype=torch.int64)

    print(dataX.shape)

    y = torch.as_tensor(y, dtype=torch.int64)

    dataset = torch.utils.data.TensorDataset(dataX, y)

    nb_classes = n_vocab

    # Neural Network

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device being used is {device}')

    # Store the loss for each epoch
    total_loss = []
    bleu_scores = []

    model = LSTMNet(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_layers'],
                    nb_classes, seq_length, device, CONFIG['dropout']).to(device)
    # model = BiLSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True)

    # training
    total_step = len(train_loader)
    start = time.time()

    for epoch in range(CONFIG['num_epochs']):
        epoch_loss = 0
        for i, (seq, lab) in enumerate(train_loader):

            # seq = seq.to(device)
            seq = seq.reshape(-1, seq_length, 1).to(device=device)
            # print(f'seq: {seq.shape}')

            # torch.cat(seq).view(len(seq), 1, -1).to(device)
            lab = lab.to(device)

            outputs = model(seq)
            # print(f'outputs: {outputs.shape}')
            # print(f'labels: {torch.max(lab, 1)[1].shape}')
            loss = loss_fn(outputs, torch.max(lab, 1)[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} ({:.2f} s)'
                      .format(epoch+1, CONFIG['num_epochs'], i+1, total_step,
                              loss.item(), time.time()-start))
        start = np.random.randint(0, len(dataX)-1)
        pattern = list(dataX[start])
        gen_text, bleu = mean_bleu(
            10, BLEU_WEIGHTS, model, pattern, seq_length, device, int_to_char, fics, character_level=False)
        bleu_scores.append(bleu)
        total_loss.append(epoch_loss / total_step)
        print(f'Loss for the epoch: {epoch_loss / total_step}')
        print(f'One BLEU score: {bleu}')

    print(f"Loss for each epoch: {total_loss}")
    print(f"One bleu score for each epoch: {bleu_scores}")

    file_name = './model-state-{}-{}-{}-{}-{}-{}-{}-{}.torch'.format(
        CONFIG['seq_length'],
        CONFIG['batch_size'],
        CONFIG['input_size'],
        CONFIG['hidden_size'],
        CONFIG['num_layers'],
        CONFIG['num_epochs'],
        CONFIG['learning_rate'],
        CONFIG['dropout']
    )

    torch.save(model.state_dict(), file_name)
