from nltk.corpus import brown
from collections import Counter

# This script generates training, validation and test sets from the
# Brown corpus in an attempt to reproduce the results by Bengio et. al. (2003)
# with FFNN language models.

def save_sentences(file_name, sentences):
    """ Saves a list of sentences to a file. Each list
    is a list containing words.
    Args:
        - file_name (str): the name of the file to be saved.
        - sentences (list): contains lists of words (str).
    Example:
    save_sentences("sentences.txt", [["Hello", "world"], ["Goodbye"]])
    """
    with open(file_name, "w") as file:
        for sent in sentences:
            for word in sent:
                file.write(" " + word)
            file.write(" \n")

# Get the word count for the original Brown corpus
words_count = Counter(brown.words())
print("Initial vocabulary size:", len(words_count))

# Words with frequency equal to or lower than 3 are replaced by <unk>
unk_symbol = "UNK"
sentences = []
new_count = Counter()
for sent in brown.sents():
    new_sent = []
    for word in sent:
        if words_count[word] > 3:
            new_sent.append(word)
        else:
            new_sent.append(unk_symbol)
    sentences.append(new_sent)
    new_count.update(new_sent)
print("Reduced vocabulary size:", len(new_count))

# Now separate into a training set containing 800k words,
# a validation set with 200k words and a test set with the rest
train_words = 800000
valid_words = 200000
train_set = []
valid_set = []
test_set = []
word_count = 0
for sent in sentences:
    if word_count < train_words:
        train_set.append(sent)
    elif train_words <= word_count and word_count < train_words + valid_words:
        valid_set.append(sent)
    else:
        test_set.append(sent)
    word_count += len(sent)

print("Training set size:", sum(map(len, train_set)), "words")
print("Validation set size:", sum(map(len, valid_set)), "words")
print("Test set size:", sum(map(len, test_set)), "words")

# Save to text files
save_sentences("brown_train.UNK.txt", train_set)
save_sentences("brown_valid.UNK.txt", valid_set)
save_sentences("brown_test.UNK.txt", test_set)
