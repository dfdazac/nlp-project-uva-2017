import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from collections import defaultdict
import pickle

CUDA = torch.cuda.is_available()

class FFNeuralModel(nn.Module):
    """ A Neural Language Model based on Bengio (2003)
    Args:
        - vocab_size (int): size of the vocabulary
        - emb_dimensions (int): word embeddings dimensions
        - context_size: number of words used as context
        - n_hidden: number of units in the hidden layer
    """
    def __init__(self, vocab_size, emb_dimensions, context_size, n_hidden):
        super(FFNeuralModel, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, emb_dimensions)
        self.linear1 = nn.Linear(emb_dimensions * context_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, vocab_size)

    def forward(self, inputs):
        """ Calculates the log-probabilities for all words
        given the inputs.
        Args:
            - inputs (tensor): (N, context_size), a tensor containing word indices
        Returns:
            - tensor: (N, vocab_size), the log-probabilities
        """
        # Get the embeddings for the inputs and reshape to row
        embeddings = self.embeddings(inputs).view((1, -1))
        # Forward propagate
        h = F.tanh(self.linear1(embeddings))
        y = self.linear2(h)
        log_probs = F.log_softmax(y)
        return log_probs

if __name__ == '__main__':
    # Load training data
    word_to_idx = defaultdict(lambda: len(word_to_idx))
    S = word_to_idx["<s>"]
    UNK = word_to_idx["<unk>"]
    def read_corpus(corpusfname):
        """ A generator that yields a list of word indices for each sentence
        in the corpus. Assumes there's one sentence per line.
        Args:
            - corpusfname (string): file name of the corpus to read
        """
        with open(corpusfname) as corpus:
            for line in corpus:
                yield [word_to_idx[word] for word in line.strip().split()]

    training_file = "../data/train_toy.txt"
    train_data = list(read_corpus(training_file))

    word_to_idx = dict(word_to_idx)
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    vocab_size = len(word_to_idx)

    pickle.dump(word_to_idx, open("word_to_idx.p", "wb"))
    pickle.dump(word_to_idx, open("idx_to_word.p", "wb"))

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

    # Since the neural model outputs log-probabilities,
    # we use the negative log-likelihood loss
    context_size = 5
    loss_function = nn.NLLLoss()
    model = FFNeuralModel(vocab_size, 60, context_size, 50)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    if CUDA:
        model.cuda()

    # Train!
    EPOCHS = 20
    print("Epoch\tLoss")
    for epoch in range(EPOCHS):
        for sentence in train_data:
            # Clear gradients
            model.zero_grad()

            sentence_loss = autograd.Variable(torch.FloatTensor([0]))
            if CUDA:
                sentence_loss = sentence_loss.cuda()

            for history, target in next_ngram_sample(sentence, context_size):
                # Forward propagate to get n-gram log-probabilities
                log_probs = model(autograd.Variable(torch.LongTensor(history)))            
                
                # Accumulate the loss for the obtained log-probability
                sentence_loss += loss_function(log_probs, autograd.Variable(torch.LongTensor([target])))

            # Backward propagate and optimize the parameters
            sentence_loss.backward()
            optimizer.step()

        print("{:d}/{:d} - {:.2f}".format(epoch+1, EPOCHS, sentence_loss.data[0]))

    torch.save(model.state_dict(), "ffnn_model.pt")
