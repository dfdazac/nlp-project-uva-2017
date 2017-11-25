import torch
import torch.autograd as autograd
from ffnn import FFNeuralModel
import pickle
import lm_eval as lme
import numpy as np
from ffnn_train import next_ngram_sample

def sentence_prob(sentence):
    """ Returns the log-probability of a sentence
    """
    sentence_idx = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in sentence.split()]
    log_probability = 0
    for history, target in next_ngram_sample(sentence_idx, context_size, word_to_idx["<s>"]):
        ngram_prob = model(autograd.Variable(torch.LongTensor(history)))
        log_probability += ngram_prob.data[0, target]
    return log_probability

model = torch.load("ffnn_model.pt")
word_to_idx = model.word_to_idx
context_size = model.context_size

print("Perplexity:", lme.perplexity("../data/train_toy.txt", sentence_prob, base=np.e))
