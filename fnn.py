from collections import defaultdict
from collections import Counter
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_preprocessing import read_embeddings
from itertools import islice, chain
import numpy as np

# load word embeddings
embeddings = read_embeddings('data/glove42B300d.txt') # np.fromstring()

# Function to read in the corpus and store
def read_dataset(filename):
    with open(filename, "r") as f:
        text = f.readlines()
        return text


def vocabulary(filename):
    V = Counter()
    with open(filename, "r") as f:
        for line in f:
            l = line.split()
            for word in l:
                V[word] =+1
    return V


# Hyper Parameters
input_size = 784
h = 50               # hidden size
context_size = 3                # order of the model/ context size
emb_dim = 300        # embedding dimensions, m, number of features associated with each word
#num_epochs = 5
#batch_size = 100
learning_rate = 0.01


# read in data
train = read_dataset('../data/train.txt')
valid = read_dataset('../data/valid.txt')
dev = read_dataset('../data/test.txt')
voc_size = len(embeddings)
embeddings_keys = list(embeddings.keys())

# Feed forward Neural Network for the Language model
class FNN(nn.Module):

    def __init__(self, n, h, o):
        super(FNN, self).__init__()
        self.linear_i = nn.Linear(n, h)
        self.linear_o = nn.Linear(h, o)

    # calculates un normalized log-probabilities for each word i
    def forward(self, x):
        z = self.linear_i(x)
        h = F.tanh(z)
        y = self.linear_o(h)
        return y

model = FNN(context_size*emb_dim, 50, voc_size)   # 300 is embedding dimension
print(model)


optimizer = optim.SGD(model.parameters(), learning_rate)

for ITER in range(100):

    train_loss = 0.0
    start = time.time()

    for i, sentence in enumerate(train):
        words = sentence.split()

        for start_i in range(len(words) - context_size):
            x = Variable(torch.from_numpy(np.concatenate([np.fromstring(embeddings[word], sep=" ") for word in words[start_i: start_i + context_size]])).double())
            y = model(x)
            loss = nn.CrossEntropyLoss()
            target = embeddings_keys.index(words[start_i + context_size])
            output = loss(y, target)
            train_loss += output.data[0]

            model.zero_grad()
            output.backward()
            # Adjust weights
            optimizer.step()

        print("iter %r: train loss/sent=%.4f, time=%.2fs" %
              (ITER, train_loss / len(train), time.time() - start))


        # perplexity

