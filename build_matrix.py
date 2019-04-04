import pickle
import re

import numpy as np

from parse_fics import Fanfic


glove = {}
fics = []
glove_path = "./glove"

# Load GloVe word vectors
with open(f'{glove_path}/6B.glove_dict.pkl', 'rb') as file:
    glove = pickle.load(file)
# Load Fanfics, 49999 in total
with open("./fics.pkl", 'rb') as file: 
    fics = pickle.load(file)

vocab = set()

for i, fic in enumerate(fics):
    for word in re.split('[^a-zA-Z]', fic.body):
        vocab.add(word)
    if(i%1000 ==0):
        print(f"Building vocabulary {i}")

target_vocab = vocab

matrix_len = len(target_vocab)
emb_dim = 50
weights_matrix = np.zeros((matrix_len, emb_dim))
word2index = {}
words_found = 0

for i, word in enumerate(target_vocab):
    try: 
        word2index[word] = i
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

print(f"Words found: {words_found}/{len(vocab)}")

with open("./weights_matrix.pkl", 'wb') as file: 
    pickle.dump(weights_matrix, file)

with open("./word2index.pkl", 'wb') as file: 
    pickle.dump(word2index, file)