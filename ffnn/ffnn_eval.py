from torch import load
from torch.nn import NLLLoss
import ffnn_utils as utils
from datetime import datetime
from numpy import e

def compute_perplexity(model, sentences):
    """ Evaluates the total perplexity of a model
    over a list of sentences. Uses batches for a
    large increase in evaluation speed.
    """
    n_tokens = 0

    # Since we want the sum of losses (i.e. log-probabilities)
    # we must disable averaging over batches
    loss_function = NLLLoss(size_average=False)
    total_log_prob = 0

    for histories, targets in utils.next_batch_ngrams(sentences, model.context_size):
        log_probs = model(utils.get_variable(histories, volatile=True))
        total_log_prob += loss_function(log_probs, utils.get_variable(targets)).data[0]
        n_tokens += len(targets)

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
        if utils.CUDA:
            model = load(model_name)
        else:
            model = load(model_name, map_location = lambda storage, loc: storage)

        sentences = utils.get_corpus_indices("../data/brown_train.txt", model.word_to_idx, model.context_size)
        train_perp = compute_perplexity(model, sentences)
        sentences = utils.get_corpus_indices("../data/brown_valid.txt", model.word_to_idx, model.context_size)
        valid_perp = compute_perplexity(model, sentences)
        sentences = utils.get_corpus_indices("../data/brown_test.txt", model.word_to_idx, model.context_size)
        test_perp = compute_perplexity(model, sentences)

        results += "{:25s}{:^10.1f}{:^10.1f}{:^10.1f}\n".format(model_name, train_perp, valid_perp, test_perp)

    now = datetime.now()
    with open("ffnn_perplexities_" + datetime.now().strftime('%Y_%m_%d_%H%M') + ".txt", "w") as file:
        file.write(results)

model_names = ["4o_30m_100h_ffnn.pt"]

evaluate_models(model_names)
