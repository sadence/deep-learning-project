import pickle

from torch import nn
import torch
import torchvision
import numpy as np
import time

from parse_fics import Fanfic

# Hyper params
seq_length = 50
batch_size = 256
input_size = 1
hidden_size = 150
num_layers = 2
num_epochs = 20
learning_rate = 0.004
dropout = 0

class LSTMNet(torch.nn.Module):

    def __init__(self,in_size,hidden_size, nb_layer, nb_classes, device, dropout):
        super(LSTMNet,self).__init__()
        self.hidden_size = hidden_size
        self.nb_layer = nb_layer
        self.nb_classes = nb_classes
        self.lstm = torch.nn.LSTM(in_size, hidden_size, nb_layer, batch_first=True, dropout = dropout)
        self.fc = torch.nn.Linear(hidden_size, nb_classes)
        self.device = device

    def forward(self,x, batch_size=batch_size):
        # initial states
        h0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(self.device)

        out,_ = self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1,:])
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
        fics = fics[:2] # begin with only this much

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

    dataX = np.array(dataX) / n_vocab
    dataX = torch.as_tensor(dataX, dtype=torch.float)

    print(dataX.shape)

    y = torch.as_tensor(y, dtype=torch.int64)

    dataset = torch.utils.data.TensorDataset(dataX, y)

    nb_classes = n_vocab

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device being used is {device}')


    model = LSTMNet(input_size, hidden_size, num_layers, nb_classes, device, dropout).to(device)
    # model = BiLSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True)


    # training
    total_step = len(train_loader)
    start = time.time()

    for epoch in range(num_epochs):
        for i, (seq, lab) in enumerate(train_loader):
            
            # seq = seq.to(device)
            seq = seq.reshape(-1, seq_length, input_size).to(device=device)

            # torch.cat(seq).view(len(seq), 1, -1).to(device)
            lab = lab.to(device)

            outputs = model(seq)
            loss = loss_fn(outputs, torch.max(lab, 1)[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} ({:.2f} s)'
                .format(epoch+1, num_epochs, i+1, total_step,
                loss.item(), time.time()-start))

    torch.save(model.state_dict(), "./model-state.torch")