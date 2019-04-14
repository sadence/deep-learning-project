import pickle
import configparser

from torch import nn
import torch
import torchtext
import numpy as np
import time

from predict import predict_bleu

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
        print(f"Building samples {j}, {len(fic.body)}")
        for i in range(0, len(fic_arr) - seq_length, 1):
            seq_in = fic_arr[i: i + seq_length]
            seq_out = fic_arr[i + seq_length]
            try:   
                x = [model.glove.stoi[word] for word in seq_in]
                y = model.glove.stoi[seq_out]
                dataX.append(x)
                dataY.append(y)
            except KeyError:
                errors+=1


    n_patters = len(dataX)
    print(f"Total patterns: {n_patters}")
    print(f'errors: {errors}')

    dataX = np.array(dataX)
    dataX = torch.as_tensor(dataX, dtype=torch.int64)

    dataY = np.array(dataY)
    dataY = torch.as_tensor(dataY, dtype=torch.int64)

    dataset = torch.utils.data.TensorDataset(dataX, dataY)

    total_loss = []
    bleu_scores = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True)

    # training
    total_step = len(train_loader)
    start = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (seq, lab) in enumerate(train_loader):

            seq = seq.reshape(-1, seq_length, 1).to(device=device)
            lab = lab.to(device)
            outputs = model(seq)
            loss = loss_fn(outputs, lab)

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
            model, pattern, seq_length, device, model.glove.itos, fics, character_level=False)
        bleu_scores.append(bleu)
        total_loss.append(epoch_loss / total_step)
        print(f'Loss for the epoch: {epoch_loss / total_step}')
        print(f'One BLEU score: {bleu}')

    print(f"Loss for each epoch: {total_loss}")
    # print(f"One bleu score for each epoch: {bleu_scores}")

    file_name = './model-word-state-{}-{}-{}-{}-{}-{}-{}-{}.torch'.format(
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


