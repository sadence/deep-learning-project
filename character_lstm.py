import pickle
import configparser

from torch import nn
import torch
import torchvision
import numpy as np
import time

from parse_fics import Fanfic
from predict import predict_bleu

# Hyper params
config = configparser.ConfigParser()
config.read('config.ini')

config = dict(config['DEFAULT'])


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


for key in config:
    config[key] = num(config[key])


class LSTMNet(torch.nn.Module):

    def __init__(self, in_size, hidden_size, nb_layer, nb_classes, device, dropout):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.nb_layer = nb_layer
        self.nb_classes = nb_classes
        self.emb = torch.nn.Embedding(nb_classes, in_size)
        self.lstm = torch.nn.LSTM(
            in_size, hidden_size, nb_layer, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, nb_classes)
        self.device = device

    def forward(self, x, batch_size=config['batch_size']):
        # print(f'x: {x.shape}')
        # -1 corresponds to batch size (but is smaller for last batch)
        x = self.emb(x).view(-1, config['seq_length'], config['input_size'])
        # initial states
        # print(f'x: {x.shape}')
        h0 = torch.zeros(self.nb_layer, x.size(
            0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.nb_layer, x.size(
            0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    def to_one_hot(i, total_classes):
        return np.eye(total_classes)[i]

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

    seq_length = config['seq_length']

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

    model = LSTMNet(config['input_size'], config['hidden_size'], config['num_layers'],
                    nb_classes, device, config['dropout']).to(device)
    # model = BiLSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=True)

    # training
    total_step = len(train_loader)
    start = time.time()

    for epoch in range(config['num_epochs']):
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
                      .format(epoch+1, config['num_epochs'], i+1, total_step,
                              loss.item(), time.time()-start))
        start = np.random.randint(0, len(dataX)-1)
        pattern = list(dataX[start])
        gen_text, bleu = predict_bleu(
            model, pattern, seq_length, device, int_to_char, fics, character_level=False)
        bleu_scores.append(bleu)
        total_loss.append(epoch_loss / total_step)
        print(f'Loss for the epoch: {epoch_loss / total_step}')
        print(f'One BLEU score: {bleu}')

    print(f"Loss for each epoch: {total_loss}")
    print(f"One bleu score for each epoch: {bleu_scores}")

    file_name = './model-state-{}-{}-{}-{}-{}-{}-{}-{}.torch'.format(
        config['seq_length'],
        config['batch_size'],
        config['input_size'],
        config['hidden_size'],
        config['num_layers'],
        config['num_epochs'],
        config['learning_rate'],
        config['dropout']
    )

    torch.save(model.state_dict(), file_name)
