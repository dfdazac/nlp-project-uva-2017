import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle
from numpy import e
from datetime import datetime

def read_dataset(filename):
    with open(filename, "r") as filename:
        data = filename.readlines()
    return data

def end_of_sent(data):
    """
    adds an end-of-sequence symbol <eos> after each sentence in the dataset
    """
    dataeos = []
    for sentence in data:
        sent = sentence.split()
        sent.append('<eos>')
        dataeos.append(sent)
    return dataeos

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation=False):
    seq_len = min(5, len(source) - 1 - i) # 5 is sequence length
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def word_to_idx(data):
    """ A generator that yields a list of word indices for each sentence
    in the data set. Assumes there's one sentence per line.
    Args:
        - Input: nested list of strings, each sublist representing a training example.
        - Output: nested list of indices, each index in a sublist represents one word in the vocabulary.
    """
    return [word2i[word] for word in data]


def batchify(data, bsz):
    # Treat training set as 1 sequence
    seq = [word for sentence in data for word in sentence]
    # Work out how cleanly we can divide the dataset into bsz parts.
    data = torch.LongTensor(seq)
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    #if torch.cuda.is_available:
     #   data = data.cuda()
    #else:
        #data = data
    return data

def evaluate_ppl(model, data_source, word2i):
    """
    Calculates perplexities given a specific model and data.
    Args: - model
          - text data in integer representation
    Out: - perplexity for the whole dataset
    """
    global ppl
    # Turn on evaluation mode which disables dropout.
    model.eval()
    criterion = nn.CrossEntropyLoss() # Returns Negative Log-Likelihood
    total_loss = 0
    ntokens = len(word2i)
    eval_batch_size = 32
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, 5):  # 5 is sequenc
        # e length which was used for the model
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
        ppl = e**(total_loss[0] / data_source.size(0))
    return ppl

def evaluate_models(model_names, word2i):
    """ Evaluates the perplexities of different neural language models
    and saves their perplexities on train, validation and test sets.
    Args:
        - model_names (list): contains the file names (str) of the
            models exported using torch.save.
    """
    results = "{:25s}{:^10s}{:^10s}{:^10s}\n".format("Model name", "Train", "Validation", "Test")

    for model_name in model_names:
        if torch.cuda.is_available():
            model = torch.load(model_name)
        else:
            model = torch.load(model_name, map_location=lambda storage, loc: storage)

        # Read in data set with EOS symbols added at the end of each sentence
        train = end_of_sent(read_dataset('./data_ptb/train.txt'))
        valid = end_of_sent(read_dataset('./data_ptb/valid.txt'))
        test = end_of_sent(read_dataset('./data_ptb/test.txt'))

        # Map words to integers
        train = [word_to_idx(train) for train in train]
        valid = [word_to_idx(valid) for valid in valid]
        test = [word_to_idx(test) for test in test]

        # Batchify to speed up evaluation
        train = batchify(train, bsz = 32)
        valid = batchify(valid, bsz = 32)
        test = batchify(test, bsz = 32)

        # Calculate Perplexities for train, test and validation
        train_perp = evaluate_ppl(model, train, word2i)
        valid_perp = evaluate_ppl(model, valid, word2i)
        test_perp = evaluate_ppl(model, test, word2i)

        results += "{:25s}{:^10f}{:^10f}{:^10f}\n".format(model_name, train_perp, valid_perp, test_perp)
        print(results)
    with open("rnn_perplexities_" + datetime.now().strftime('%Y_%m_%d_%H%M') + ".txt", "w") as file:
        file.write(results)

# Insert names of models to be evaluated ...
model_names = ["model.pt"]

word2i = pickle.load(open('word2i','rb'))
evaluate_models(model_names, word2i)
