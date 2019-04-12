import torch

from parse_fics import Fanfic
from character_lstm import LSTMNet

import numpy as np
import pickle
import sys

# Hyper params
seq_length = 200
batch_size = 100
input_size = 80
hidden_size = 100
num_layers = 2
num_epochs = 1
learning_rate = 0.01
dropout = 0
nb_classes = 74


model = LSTMNet(input_size, hidden_size, num_layers,
                nb_classes, 'cpu', dropout)
model.load_state_dict(torch.load("./model-state.torch"))
model.eval()


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

with torch.no_grad():
    # generate characters
    for i in range(1000):
        x = np.reshape(pattern, (-1, seq_length, 1))
        x = torch.as_tensor(x, dtype=torch.int64)
        out = model(x).view(nb_classes)
        # index = np.argmax(out).item() # read value of 1d tensor
        # print(out.shape)
        # print(out)
        # top_indexes = torch.topk(out, 5, largest=True)
        probs = torch.nn.functional.softmax(out, 0)
        # index = np.random.choice(top_indexes[1])
        index = np.random.choice(np.arange(0, nb_classes), p=probs.numpy())
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")
