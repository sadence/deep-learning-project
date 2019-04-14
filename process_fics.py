import re
import pickle

from parse_fics import Fanfic

fics = []

with open("./fics.pkl", 'rb') as file:
    fics = pickle.load(file)

reg = re.compile('([^a-zA-Z\ ])')

for fic in fics:
    fic.body = re.sub(reg, r' \1 ', fic.body).lower()

with open("./fics-processed.pkl", 'wb') as file:
    pickle.dump(fics, file)