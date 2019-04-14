import torch

from parse_fics import Fanfic
from bleu import compute_bleu

import numpy as np
import pickle
import configparser
import sys
from statistics import mean 


BLEU_WEIGHTS = [0.5, 0.3, 0.2, 0]

def to_one_hot(i, total_classes):
    return np.eye(total_classes)[i]

def mean_bleu(n, weights, model, seq_length, device, int_to_char, char_to_int, fics, character_level=False):
    n_chars = 0
    n_vocab = len(char_to_int)
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

    dataX = np.array(dataX)
    # dataX = torch.as_tensor(dataX, dtype=torch.float)

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
        generated_text = ''.join(generated_text)
        return generated_text, compute_bleu(weights, fics, generated_text, character_level)


if __name__ == "__main__":
    from character_lstm import LSTMNet

    config = configparser.ConfigParser()
    config.read('config.ini')

    config = config['DEFAULT']

    seq_length = int(config['seq_length'])
    batch_size = int(config['batch_size'])
    input_size = int(config['input_size'])
    hidden_size = int(config['hidden_size'])
    num_layers = int(config['num_layers'])
    num_epochs = int(config['num_epochs'])
    learning_rate = float(config['learning_rate'])
    dropout = float(config['dropout'])

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
    # dataX = torch.as_tensor(dataX, dtype=torch.float)

    start = np.random.randint(0, len(dataX)-1)
    pattern = list(dataX[start])
    print("Seed:")
    print(''.join([int_to_char[value] for value in pattern]))

    model = LSTMNet(input_size, hidden_size, num_layers,
                    n_vocab, seq_length, 'cpu', dropout)

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

    # open on cpu
    model.load_state_dict(torch.load(
        file_name, map_location=lambda storage, loc: storage))
    model.eval()

    with torch.no_grad():
        print(mean_bleu(10, BLEU_WEIGHTS, model, seq_length, 'cpu', int_to_char, char_to_int, fics))
