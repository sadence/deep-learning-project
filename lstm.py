import pickle

import torch
import numpy as np

from torchvision import datasets

from parse_fics import Fanfic

fics = []

# Load Fanfics, 49999 in total
with open("./fics.pkl", 'rb') as file: 
    fics = pickle.load(file)

with open("./weights_matrix.pkl", 'rb') as file: 
    weights_matrix = pickle.load(file)

# link between the word and the matrix index
with open("./word2index.pkl", 'rb') as file: 
    word2index = pickle.load(file)


# Hyper-parameters
train_dataset_size = 40000
batch_size = 100
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_epochs = 2
learning_rate = 0.001
nb_classes = len(weights_matrix)

# Set cuda as device if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device being used is {device}')

# Load our fics as datasets 
train_loader = torch.utils.data.DataLoader(dataset=fics[:train_dataset_size], batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=fics[train_dataset_size:], batch_size=batch_size, shuffle=False)

# define LSTM model
class LSTMNet(torch.nn.Module):

    def __init__(self,in_size,hidden_size, nb_layer, nb_classes):
        super(LSTMNet,self).__init__()
        self.hidden_size = hidden_size
        self.nb_layer = nb_layer
        self.nb_classes = nb_classes
        self.embedding_layer, num_embeddings, embedding_dim = self.create_emb_layer(weights_matrix)
        self.lstm = torch.nn.LSTM(in_size,hidden_size,nb_layer,batch_first=True)
        self.fc = torch.nn.Linear(hidden_size,nb_classes)

    def create_emb_layer(self, weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim


    def forward(self,x):
        # initial states
        h0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(device)

        out,_ = self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        return out


model = LSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)
# model = BiLSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
loss_fn = nn.CrossEntropyLoss()

# training
total_step = len(train_loader)
start = time.time()
for epoch in range(num_epochs):
    for i,fic in enumerate(train_loader):
        img = img.reshape(-1,sequence_length,input_size).to(device)
        lab = lab.to(device)

        outputs = model(img)
        loss = loss_fn(outputs,lab)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} ({:.2f} s)'
            .format(epoch+1, num_epochs, i+1, total_step,
            loss.item(), time.time()-start))

# test
with torch.no_grad():
    correct = 0
    total = 0
    for img, lab in test_loader:
        img = img.reshape(-1,sequence_length,input_size).to(device)
        lab = lab.to(device)
        outputs = model(img)
        _, pred = torch.max(outputs.data,1)
        total += lab.size(0)
        correct += (pred == lab).sum().item()

    print('Test Accuracy: {}%'.format(100. * correct / total) )
