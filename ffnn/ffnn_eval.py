import torch
import torch.autograd as autograd
from ffnn_training import FFNeuralModel
import pickle
import lm_eval as lme
import numpy as np

def next_ngram_sample(sentence, context_size):
        # Start with a history of sentence delimiters
        history = [S] * context_size
        # Add a sentence delimiter at the end
        delim_sentence = sentence + [S]
        # Move a window of the size of the context and yield history, target
        for i in range(len(delim_sentence)):
            target = delim_sentence[i]
            yield history, target
            history = history[1:] + [target]

def sentence_prob(sentence):
    """ Returns the log-probability of a sentence
    """
    sentence_idx = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in sentence.split()]
    log_probability = 0
    for history, target in next_ngram_sample(sentence_idx, context_size):
        ngram_prob = model(autograd.Variable(torch.LongTensor(history)))
        log_probability += ngram_prob.data[0, target]
    return log_probability
        
word_to_idx = pickle.load(open("word_to_idx.p", "rb"))
idx_to_word = pickle.load(open("idx_to_word.p", "rb"))
S = word_to_idx["<s>"]

context_size = 5
model = FFNeuralModel(vocab_size=len(word_to_idx),
    emb_dimensions=60, context_size=context_size, n_hidden=50)
    
model.load_state_dict(torch.load("ffnn_model.pt",
    map_location=lambda storage, loc:storage))

print(lme.perplexity("../data/train_toy.txt", sentence_prob, base=np.e))