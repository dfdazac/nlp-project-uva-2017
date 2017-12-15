import torch.nn as nn
import torch.nn.functional as F

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
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, vocab_size)
        self.drop = nn.Dropout(0.5)

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
        embeddings = self.drop(self.embeddings(inputs).view(len(inputs), -1))
        # Forward propagate
        h1 = F.relu(self.linear1(embeddings))
        h2 = F.relu(self.linear2(h1))
        y = self.drop(self.linear3(h2))
        log_probs = F.log_softmax(y)
        return log_probs
