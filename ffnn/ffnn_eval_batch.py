import torch
import torch.nn as nn
import ffnn_train as nnt
from numpy import e

def batch_perplexity(model, sentences):
    """ Evaluates the total perplexity of a model
    over a list of sentences. Uses batches for a
    large increase in evaluation speed.
    """

    # When counting tokens, add 1 to all sentences
    # to count the end of sentence symbol
    n_tokens = sum(map(lambda s: len(s) + 1, sentences))

    context_size = model.context_size
    S = model.word_to_idx["<s>"]

    # Since we want the sum of losses (i.e. log-probabilities)
    # we must disable averaging over batches
    loss_function = nn.NLLLoss(size_average=False)
    total_log_prob = 0

    for histories, targets in nnt.next_batch(sentences, context_size, len(sentences), S):
        log_probs = model(nnt.get_variable(histories))
        total_log_prob += loss_function(log_probs, nnt.get_variable(targets)).data[0]

    # Note that NLLLoss returns the **negative** log-likelihood,
    # so the minus sign is dropped in the perplexity calculation
    return e ** (total_log_prob/n_tokens)

model_name = "4o_30m_50h_ffnn.pt"
if nnt.CUDA:
    model = torch.load(model_name)
else:
    model = torch.load(model_name, map_location = lambda storage, loc: storage)

sentences = nnt.get_corpus_indices("../data/train.txt", model.word_to_idx)
print("Perplexity:", batch_perplexity(model, sentences))
