from torch import save
from torch.nn import NLLLoss
from torch.optim import Adam
from ffnn import FFNeuralModel
import ffnn_utils as utils

def train(train_fname, valid_fname, context_size, emb_dimensions, n_hidden):
    train_data, word_to_idx = utils.read_corpus_data(train_fname, context_size)
    valid_data = utils.get_corpus_indices(valid_fname, word_to_idx, context_size)

    # Setup model and training
    model = FFNeuralModel(emb_dimensions, context_size, n_hidden, word_to_idx)
    if utils.CUDA:
        model.cuda()
    loss_function = NLLLoss()
    valid_loss_function = NLLLoss(size_average=False)
    optimizer = Adam(model.parameters(), weight_decay=0.0001)
    batch_size = 8
    epochs = 25

    # Use the settings for the model file name
    model_fname = "{:d}o_{:d}m_{:d}h_ffnn.pt".format(context_size, emb_dimensions, n_hidden)

    print("Training model with context size {:d}, embedding dimensions {:d} and {:d} hidden layers.".format(
        context_size, emb_dimensions, n_hidden))
    print("{:6s}  {:^23s}".format("", "Loss"))
    print("{:6s}  {:^11s}  {:^11s}".format("Epoch", "Train", "Validation"))

    # Keep track of the previous loss for early termination
    prev_valid_batch_loss = float("inf")
    terminate_early = False
    for epoch in range(epochs):
        batch_train_loss = 0
        batch_valid_loss = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            # To each sentence add sentence delimiters
            model.zero_grad()
            for histories, targets in utils.next_batch_ngrams(batch, context_size):
                # Predict
                log_probs = model(utils.get_variable(histories))
                train_loss = loss_function(log_probs, utils.get_variable(targets))
                train_loss.backward()

                batch_train_loss += train_loss.data[0]
            optimizer.step()

        # Evaluate on validation set
        for histories, targets in utils.next_batch_ngrams(valid_data, context_size):
            # Predict
            log_probs = model(utils.get_variable(histories, volatile=True))
            # Evaluate loss
            batch_valid_loss += valid_loss_function(log_probs, utils.get_variable(targets)).data[0]

        # If validation loss decreased, save model
        if batch_valid_loss <= prev_valid_batch_loss:
            prev_valid_batch_loss = batch_valid_loss
            save(model, model_fname)
        else:
            # Early termination
            terminate_early = True
            print("Terminating due to increase in validation loss:")
        #print("{:2d}/{:2d}:  {:^10.1f}  {:^10.1f}".format(epoch+1, epochs, batch_train_loss, batch_valid_loss))
        print("{:2d}/{:2d}:  {:^11.1f}  {:^11.1f}".format(epoch+1, epochs, batch_train_loss, batch_valid_loss))

        if terminate_early:
            break

    print("Saved best model on validation set as", model_fname)

if __name__ == '__main__':
    print("Loading data...")
    train_fname = "../data/brown_train.txt"
    valid_fname = "../data/brown_valid.txt"

    train(train_fname, valid_fname, context_size=4, emb_dimensions=30, n_hidden=100)
