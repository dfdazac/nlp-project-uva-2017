from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from ffnn import FFNeuralModel

CUDA = torch.cuda.is_available()

def read_corpus(corpusfname, word_to_idx):
    """ A generator that yields a list of word indices for each sentence
    in the corpus. Assumes there's one sentence per line.
    Args:
        - corpusfname (string): file name of the corpus to read
    """
    with open(corpusfname) as corpus:
        for line in corpus:
            yield [word_to_idx[word] for word in line.strip().split()]

def next_ngram_sample(sentence, context_size, S):
    """ Generates tuples (history, target) from a window
    of lenght context_size that moves across the sentence.
    Args:
        - sentence (list): contains indices of words (int) in a sentence.
        - context_size (int): the number of words use as context for the next
        - S (int): the index of the sentence delimiter <s>
    """    
    # Start with a history of sentence delimiters
    history = [S] * context_size
    # Add a sentence delimiter at the end
    delim_sentence = sentence + [S]
    # Move a window of the size of the context and yield history, target
    for i in range(len(delim_sentence)):
        target = delim_sentence[i]
        yield history, target
        history = history[1:] + [target]

def get_variable_tensor(x, dtype="long"):
    """ Helper function to get autograd.Variable given an array.
    The associated torch tensor can be of type long or float.
    Args:
        - x (array): the array to be converted.
        - dtype ({"long", "float"}, optional): The type of the torch tensor.
    """
    if dtype == "long":
        tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    elif dtype == "float":
        tensor = torch.cuda.FloatTensor(x) if CUDA else torch.FloatTensor(x)
    else:
        raise ValueError("Invalid tensor type")
    return autograd.Variable(tensor)

def next_batch(data, context_size, batch_size, S):
    """ Generates minibatches of histories and targets.
    Args:
        - data (list): contains lists of words (int), each list
            represents a sentence.
        - context_size (int): the number of words used as context.
        - batch_size (int): the number of histories and targets
            per minibatch.
        - S (int): the index of the sentence delimiter <s>.
    """
    start_padding = [S] * context_size

    for i in range(0, len(data), batch_size):
        # To each sentence add sentence delimiters
        batch = [start_padding + d + [S] for d in data[i:i+batch_size]]

        max_length = max(map(len, batch))

        # For each minibatch generate histories of lenght context_size
        # as well as targets
        for j in range(max_length - context_size):
            histories = []
            targets = []

            # The difference of length between the sentences is handled
            # by only generating history - target pairs when it is possible
            for sentence in batch:
                if j < len(sentence) - context_size:
                    histories.append(sentence[j:j + context_size])
                    targets.append(sentence[j + context_size])
            yield histories, targets


if __name__ == '__main__':
    # Load training data
    word_to_idx = defaultdict(lambda: len(word_to_idx))
    S = word_to_idx["<s>"]
    UNK = word_to_idx["<unk>"]
    training_file = "../data/train.txt"
    train_data = list(read_corpus(training_file, word_to_idx))
    word_to_idx = dict(word_to_idx)
    
    # Setup model
    emb_dimensions = 60
    context_size = 4
    n_hidden = 50
    model = FFNeuralModel(emb_dimensions, context_size, n_hidden, word_to_idx)
    if CUDA:
        model.cuda()

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    batch_size = 20

    try:
        # Train!
        EPOCHS = 20
        print("Epoch\tLoss")
        for epoch in range(EPOCHS):
            for histories, targets in next_batch(train_data, context_size, batch_size, S):
                # Clear gradients
                model.zero_grad()

                loss = get_variable_tensor([0], dtype="float")                
                log_probs = model(get_variable_tensor(histories))
                    
                # Accumulate the loss for the obtained log-probability
                loss = loss_function(log_probs, get_variable_tensor(targets))

                # Backward propagate and optimize the parameters
                loss.backward()
                optimizer.step()

            print("{:d}/{:d} - {:.2f}".format(epoch+1, EPOCHS, loss.data[0]))
    finally:
        torch.save(model, "ffnn_model.pt")
