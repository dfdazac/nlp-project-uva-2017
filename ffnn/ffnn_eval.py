import torch
import torch.autograd as autograd
from numpy import e
from ffnn_train import FFNeuralModel, next_ngram_sample, get_variable

def perplexity(corpusfname, scoringfcn, base=10):
    """ Evaluates the perplexity of a language model
    on a given corpus. Assumes there is one sentence per line.
    Args:
        - corpusfname (str): the corpus file name
        - scoringfcn (function): it should take a sentence (str).
                      as an argument and return its log-base probability (float).
        - base (int): the base of the log-probability returned by the model
    Returns:
        - float: the perplexity of the model on the corpus.
    """
    n_words = 0
    log_p_sum = 0
    with open(corpusfname) as corpus:
        for line in corpus:
            log_p_sum += scoringfcn(line)
            # Add 1 for the end of sentence symbol
            n_words += len(line.split()) + 1

    return base ** (-log_p_sum/n_words)


def sentence_prob(sentence):
    """ Returns the log-probability of a sentence
    """
    sentence_idx = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in sentence.split()]
    log_probability = 0
    for history, target in next_ngram_sample(sentence_idx, context_size, word_to_idx["<s>"]):
        ngram_prob = model(get_variable(history))
        log_probability += ngram_prob.data[0, target]
    return log_probability

model = torch.load("4o_30m_100h_ffnn.pt", map_location = lambda storage, loc: storage)
word_to_idx = model.word_to_idx
context_size = model.context_size

print("Perplexity:", perplexity("../data/train.txt", sentence_prob, base=e))
