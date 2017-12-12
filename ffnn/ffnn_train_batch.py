from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from ffnn import FFNeuralModel
from numpy import e

torch.manual_seed(42)

CUDA = torch.cuda.is_available()
EOS_SYMBOL = "<s>"
UNK_SYMBOL = "<unk>"

def read_corpus_data(corpusfname):
    """ A generator that yields a list of word indices for each sentence
    in the corpus. Assumes there's one sentence per line.
    Args:
        - corpusfname (string): file name of the corpus to read
    Returns:
        - list: a list of list of indices (int), one list per sentence.
        - dict: maps word to indices obtained from the corpus
    """
    word_to_idx = defaultdict(lambda: len(word_to_idx))
    S = word_to_idx[EOS_SYMBOL]
    UNK = word_to_idx[UNK_SYMBOL]

    sentences = []

    with open(corpusfname) as corpus:
        for line in corpus:
            sentences.append([word_to_idx[word] for word in line.strip().split()])

    word_to_idx = dict(word_to_idx)
    return sentences, word_to_idx

def get_corpus_indices(corpusfname, word_to_idx):
    """ Reads corpus data given an existing word to index
    dictionary. Useful to get the indices of a validation
    corpus from a dictionary obtained during training.
    Args:
        - corpusfname (string): file name of the corpus to read
        - word_to_idx (dict): maps word to indices
    Returns:
        - list: a list of list of indices (int), one list per sentence.
    """
    sentences = []
    with open(corpusfname) as corpus:
        for line in corpus:
            sentences.append([word_to_idx.get(word, word_to_idx["<unk>"]) for word in line.strip().split()])
    return sentences

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

def get_variable(x, volatile=False):
    """ Helper function to get autograd.Variable given an array.
    The array is converted to torch.LongTensor.
    Args:
        - x (array): the array to be converted.
    """
    tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return autograd.Variable(tensor, volatile=volatile)

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

def next_batch_ngrams(batch, context_size):
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


def train(train_data, valid_data, word_to_idx, context_size, emb_dimensions, n_hidden):
    # Count tokens including end of sentence symbol
    n_tokens_train = sum(map(lambda s: len(s) + 1, train_data))
    n_tokens_valid = sum(map(lambda s: len(s) + 1, valid_data))
    print("Training tokens:", n_tokens_train)
    print("Validation tokens:", n_tokens_valid)
    S = word_to_idx[EOS_SYMBOL]
    start_padding = [S] * context_size

    # Setup model
    model = FFNeuralModel(emb_dimensions, context_size, n_hidden, word_to_idx)
    if CUDA:
        model.cuda()
    # Setup training
    loss_function = nn.NLLLoss()
    valid_loss_function = nn.NLLLoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)
    batch_size = 16
    epochs = 25

    # Use the settings for the model file name
    model_fname = "{:d}o_{:d}m_{:d}h_ffnn.pt".format(context_size, emb_dimensions, n_hidden)

    print("Training model with context size {:d}, embedding dimensions {:d} and {:d} hidden layers.".format(
        context_size, emb_dimensions, n_hidden))
    print("{:6s}  {:^22s}".format("", "Loss"))
    print("{:6s}  {:^10s}  {:^10s}".format("Epoch", "Train", "Validation"))

    # Keep track of the previous loss for early termination
    prev_valid_batch_loss = float("inf")
    terminate_early = False
    for epoch in range(epochs):
        batch_train_loss = 0
        batch_valid_loss = 0

        for i in range(0, len(train_data), batch_size):
            # To each sentence add sentence delimiters
            batch = [start_padding + d + [S] for d in train_data[i:i+batch_size]]
            model.zero_grad()
            for histories, targets in next_batch_ngrams(batch, context_size):
                # Predict
                log_probs = model(get_variable(histories))
                train_loss = loss_function(log_probs, get_variable(targets))
                train_loss.backward()

                batch_train_loss += train_loss.data[0]
                optimizer.step()

        # Evaluate on validation set
        # TODO: Use next_batch_ngrams instead of next_batch here, and remove next_batch
        for histories, targets in next_batch(valid_data, context_size, len(valid_data), word_to_idx[EOS_SYMBOL]):
            # Predict
            log_probs = model(get_variable(histories, volatile=True))
            # Evaluate loss
            batch_valid_loss += valid_loss_function(log_probs, get_variable(targets)).data[0]

        # If validation loss decreased, save model
        if batch_valid_loss <= prev_valid_batch_loss:
            prev_valid_batch_loss = batch_valid_loss
            torch.save(model, model_fname)
        else:
            # Early termination
            terminate_early = True
            print("Terminating due to increase in validation loss:")
        #print("{:2d}/{:2d}:  {:^10.1f}  {:^10.1f}".format(epoch+1, epochs, batch_train_loss, batch_valid_loss))
        print("{:2d}/{:2d}:  {:.9f}  {:.9f}".format(epoch+1, epochs, batch_train_loss, batch_valid_loss))

        if terminate_early:
            break

    print("Saved best model on validation set as", model_fname)

if __name__ == '__main__':
    print("Loading data...")
    training_file = "../data/brown_train.txt"
    validation_file = "../data/brown_valid.txt"
    train_data, word_to_idx = read_corpus_data(training_file)
    valid_data = get_corpus_indices(validation_file, word_to_idx)

    train(train_data, valid_data, word_to_idx, context_size=4, emb_dimensions=30, n_hidden=100)
