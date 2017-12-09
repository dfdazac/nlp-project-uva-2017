import torch
import torch.nn as nn
import ffnn_train as nnt
from numpy import e
from datetime import datetime

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

def evaluate_models(model_names):
    """ Evaluates the perplexities of different neural language models
    and saves their perplexities on train, validation and test sets.
    Args:
        - model_names (list): contains the file names (str) of the
            models exported using torch.save.
    """
    results = "{:25s}{:^10s}{:^10s}{:^10s}\n".format("Model name", "Train", "Validation", "Test")

    for model_name in model_names:
        if nnt.CUDA:
            model = torch.load(model_name)
        else:
            model = torch.load(model_name, map_location = lambda storage, loc: storage)

        sentences = nnt.get_corpus_indices("../data/train.txt", model.word_to_idx)
        train_perp = int(batch_perplexity(model, sentences))
        sentences = nnt.get_corpus_indices("../data/valid.txt", model.word_to_idx)
        valid_perp = int(batch_perplexity(model, sentences))
        sentences = nnt.get_corpus_indices("../data/test.txt", model.word_to_idx)
        test_perp = int(batch_perplexity(model, sentences))

        results += "{:25s}{:^10d}{:^10d}{:^10d}\n".format(model_name, train_perp, valid_perp, test_perp)

    now = datetime.now()
    with open("ffnn_perplexities_" + datetime.now().strftime('%Y_%m_%d_%H%M') + ".txt", "w") as file:
        file.write(results)

model_names = ["4o_30m_100h_ffnn.pt"]

evaluate_models(model_names)
