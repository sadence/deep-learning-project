import torch

from parse_fics import Fanfic
from bleu import compute_bleu

import numpy as np
import pickle
import configparser
import sys
from statistics import mean 

BLEU_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

def mean_bleu(n, weights, model, seq_length, device, int_to_char, fics, character_level=False):
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
                print(seq_in)
                errors+=1


    n_patters = len(dataX)
    print(f"Total patterns: {n_patters}")
    print(f'errors: {errors}')

    dataX = np.array(dataX)

    bleus = []
    for _ in range(0, n):
        start = np.random.randint(0, len(dataX)-1)
        pattern = list(dataX[start])
        _, bleu = predict_bleu(
            weights, model, pattern, seq_length, device, int_to_char, fics, character_level=False)
        bleus.append(bleu)
    return mean(bleus)

def predict_bleu(weights, model, pattern, seq_length, device, int_to_char, fics, character_level=False):
    """
    Generate text and compute BLEU
    """
    nb_classes = len(int_to_char)
    with torch.no_grad():
        generated_text = []
        for i in range(1000):
            x = np.reshape(pattern, (-1, seq_length, 1))
            x = torch.as_tensor(x, dtype=torch.int64).to(device=device)
            out = model(x).view(nb_classes)
            # index = np.argmax(out).item() # read value of 1d tensor
            # print(out.shape)
            # print(out)
            # top_indexes = torch.topk(out, 5, largest=True)
            probs = torch.nn.functional.softmax(out, 0)
            # index = np.random.choice(top_indexes[1])
            index = np.random.choice(
                np.arange(0, nb_classes), p=probs.to(device='cpu').numpy())
            result = int_to_char[index]
            generated_text.append(result)
            seq_in = [int_to_char[value.item()] for value in pattern]
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        generated_text = ' '.join(generated_text)
        print(generated_text)
        return generated_text, compute_bleu(weights, fics, generated_text, character_level)



if __name__ == "__main__":
    from word_lstm import LSTMWordNet, BengioNet

    config = configparser.ConfigParser()
    config.read("config.ini")

    if sys.argv[1] == "bengio":
        config = config["BENGIO"]
    else:
        config = config["WORD-LSTM"]


    seq_length = int(config["seq_length"])
    batch_size = int(config["batch_size"])
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_layers"])
    num_epochs = int(config["num_epochs"])
    learning_rate = float(config["learning_rate"])
    dropout = float(config["dropout"])


    def to_one_hot(i, total_classes):
        return np.eye(total_classes)[i]

    fics = []

    # Load Fanfics, 49999 in total
    with open("./fics-processed.pkl", "rb") as file:
        fics = pickle.load(file)
        fics = fics[:3]  # begin with only this much

    device = 'cpu'
    print(f'Device being used is {device}')

    total_loss = []
    bleu_scores = []

    if len(sys.argv) > 1 and sys.argv[1] == "bengio":
        model = BengioNet(hidden_size, num_layers, device, dropout).to(device)
    else:
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
                print(seq_in)
                errors+=1


    n_patters = len(dataX)
    print(f"Total patterns: {n_patters}")
    print(f'errors: {errors}')

    dataX = np.array(dataX)


    start = np.random.randint(0, len(dataX)-1)
    pattern = list(dataX[start])
    print("Seed:")
    print(' '.join([model.glove.itos[value] for value in pattern]))

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

    # open on cpu
    model.load_state_dict(torch.load(
        file_name, map_location=lambda storage, loc: storage))
    model.eval()

    nb_classes = model.nb_classes

    with torch.no_grad():
        print(mean_bleu(10, BLEU_WEIGHTS, model, seq_length, 'cpu', model.glove.itos, fics))
