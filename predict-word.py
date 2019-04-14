import torch

from parse_fics import Fanfic

import numpy as np
import pickle
import configparser
import sys


if __name__ == "__main__":
    from word_lstm import LSTMWordNet

    config = configparser.ConfigParser()
    config.read("config.ini")

    config = config["WORD-LSTM"]

    seq_length = int(config["seq_length"])
    batch_size = int(config["batch_size"])
    input_size = 300
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
        fics = [fics[4]]  # begin with only this much

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device being used is {device}')

    total_loss = []
    bleu_scores = []

    model = LSTMWordNet(input_size, hidden_size, num_layers, device, dropout).to(device)

    
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
    print(''.join([model.glove.stoi[value] for value in pattern]))

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

    with torch.no_grad():
        # generate characters
        generated_text = []
        for i in range(1000):
            x = np.reshape(pattern, (-1, seq_length, 1))
            x = torch.as_tensor(x, dtype=torch.int64)
            out = model(x).view(n_vocab)
            probs = torch.nn.functional.softmax(out, 0)
            index = np.random.choice(np.arange(0, model.nb_classes), p=probs.numpy())
            result = model.glove.itos[index]
            generated_text.append(result)
            seq_in = [model.glove.itos[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        print("\nDone.")
        generated_text = ''.join(generated_text)
