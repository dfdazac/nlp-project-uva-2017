from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from numpy import e

class FFNeuralModel(nn.Module):
    """ A Neural Language Model based on Bengio (2003)
    Args:
        - emb_dimensions (int): word embeddings dimensions
        - context_size: number of words used as context
        - n_hidden: number of units in the hidden layer
        - word_to_idx (dict): a dictionary of word indices.
            It allows to make meaningful predictions based on the
            data used during training.
    """
    def __init__(self, emb_dimensions, context_size, n_hidden, word_to_idx):
        super(FFNeuralModel, self).__init__()

        vocab_size = len(word_to_idx)
        self.embeddings = nn.Embedding(vocab_size, emb_dimensions)
        self.linear1 = nn.Linear(emb_dimensions * context_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, vocab_size)

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: word for word, i in word_to_idx.items()}
        self.context_size = context_size

    def forward(self, inputs):
        """ Calculates the log-probabilities for all words
        given the inputs.
        Args:
            - inputs (tensor): (N, context_size), a tensor containing word indices
        Returns:
            - tensor: (N, vocab_size), the log-probabilities
        """
        # Get the embeddings for the inputs and reshape to N rows
        embeddings = self.embeddings(inputs).view((1, -1))
        # Forward propagate
        h = F.tanh(self.linear1(embeddings))
        y = self.linear2(h)
        log_probs = F.log_softmax(y)
        return log_probs


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

def train(train_data, valid_data, word_to_idx, context_size, emb_dimensions, n_hidden):
    # Count tokens including end of sentence symbol
    n_tokens_train = sum(map(lambda s: len(s) + 1, train_data))
    n_tokens_valid = sum(map(lambda s: len(s) + 1, valid_data))

    S = word_to_idx[EOS_SYMBOL]

    # Setup model
    model = FFNeuralModel(emb_dimensions, context_size, n_hidden, word_to_idx)
    if CUDA:
        model.cuda()
    # Setup training
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)
    epochs = 25

    # Use the settings for the model file name
    model_fname = "{:d}o_{:d}m_{:d}h_ffnn.pt".format(context_size, emb_dimensions, n_hidden)

    print("Training model with context size {:d}, embedding dimensions {:d} and {:d} hidden layers.".format(
        context_size, emb_dimensions, n_hidden))
    print("{:6s}  {:^22s}".format("", "Loss"))
    print("{:6s}  {:^10s}  {:^10s}".format("Epoch", "Train", "Validation"))

    # Keep track of the previous loss for early termination
    prev_valid_loss = float("inf")
    terminate_early = False
    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0

        for sentence in train_data:
            # Clear gradients
            model.zero_grad()

            sentence_loss = autograd.Variable(torch.cuda.FloatTensor([0])) if CUDA else autograd.Variable(torch.FloatTensor([0]))
            for history, target in next_ngram_sample(sentence, context_size, S):
                # Forward propagate to get n-gram log-probabilities
                log_probs = model(get_variable(history))

                # Accumulate the loss for the obtained log-probability
                sentence_loss += loss_function(log_probs, get_variable([target]))

            train_loss += sentence_loss.data[0]
            # Backward propagate and optimize the parameters
            sentence_loss.backward()
            optimizer.step()

        # Check loss in validation set
        for sentence in valid_data:
            for history, target in next_ngram_sample(sentence, context_size, S):
                log_probs = model(get_variable(history, volatile=True))
                valid_loss += loss_function(log_probs, get_variable([target])).data[0]

        # If validation loss decreased, save model
        if valid_loss <= prev_valid_loss:
            prev_valid_loss = valid_loss
            torch.save(model, model_fname)
        else:
            # Early termination
            terminate_early = True
            print("Terminating due to increase in validation loss:")

        print("{:2d}/{:2d}:  {:.9f}  {:.9f}".format(epoch+1, epochs, train_loss, valid_loss))

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
