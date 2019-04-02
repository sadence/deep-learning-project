import bcolz
import pickle

glove_path = "./glove.1"

vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

pickle.dump(glove, open(f'{glove_path}/6B.glove_dict.pkl', 'wb'))