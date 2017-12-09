import torch
import torch.autograd as autograd
from ffnn import FFNeuralModel
import lm_eval as lme
from numpy import e
from ffnn_train import next_ngram_sample

def sentence_prob(sentence):
    """ Returns the log-probability of a sentence
    """
    sentence_idx = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in sentence.split()]
    log_probability = 0
    for history, target in next_ngram_sample(sentence_idx, context_size, word_to_idx["<s>"]):
        ngram_prob = model(autograd.Variable(torch.LongTensor([history])))
        log_probability += ngram_prob.data[0, target]
    return log_probability

model = torch.load("4o_30m_100h_ffnn.pt", map_location = lambda storage, loc: storage)
word_to_idx = model.word_to_idx
context_size = model.context_size

print("Perplexity:", lme.perplexity("../data/train.txt", sentence_prob, base=e))
