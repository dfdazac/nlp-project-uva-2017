from collections import defaultdict
import torch
import torch.autograd as autograd

CUDA = torch.cuda.is_available()
EOS_SYMBOL = "<s>"
UNK_SYMBOL = "<unk>"

def read_corpus_data(corpusfname, context_size):
    """ A generator that yields a list of word indices for each sentence
    in the corpus. Assumes there's one sentence per line. To each sentence
    it adds start and end of sentence delimiters according to context_size.
    Args:
        - corpusfname (string): file name of the corpus to read
        - context_size (int): the number of sentence delimiters (<s>)
        added to each sentence. This is used when training a language model.
    Returns:
        - list: a list of list of indices (int), one list per sentence.
        - dict: maps word to indices obtained from the corpus
    """
    # Create word to index dictionary and register sentence delimiter and unknown
    # symbols
    word_to_idx = defaultdict(lambda: len(word_to_idx))
    eos_idx = word_to_idx[EOS_SYMBOL]
    UNK = word_to_idx[UNK_SYMBOL]

    start_padding = [eos_idx] * context_size
    sentences = []

    with open(corpusfname) as corpus:
        for line in corpus:
            # Add the sentence with start and end of sentence delimiters
            sentences.append(start_padding + [word_to_idx[word] for word in line.strip().split()] + [eos_idx])

    # Close the dictionary
    word_to_idx = dict(word_to_idx)
    return sentences, word_to_idx

def get_corpus_indices(corpusfname, word_to_idx, context_size):
    """ Reads corpus data given an existing word to index
    dictionary. Useful to get the indices of a validation
    corpus from a dictionary obtained during training.
    Args:
        - corpusfname (string): file name of the corpus to read
        - word_to_idx (dict): maps word to indices
    Returns:
        - list: a list of list of indices (int), one list per sentence.
    """
    eos_idx = word_to_idx[EOS_SYMBOL]
    start_padding = [eos_idx] * context_size
    sentences = []
    with open(corpusfname) as corpus:
        for line in corpus:
            sentences.append(start_padding + [word_to_idx.get(word, word_to_idx["<unk>"]) for word in line.strip().split()] + [eos_idx])
    return sentences

def get_variable(x, volatile=False):
    """ Helper function to get autograd.Variable given an array.
    The array is converted to torch.LongTensor.
    Args:
        - x (array): the array to be converted.
    """
    tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return autograd.Variable(tensor, volatile=volatile)

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
