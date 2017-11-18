from collections import OrderedDict

def read_embeddings(emb_fname):
    """ Reads from a text file containing word embeddings.
    Returns a dictionary where the key is a word (string)
    and the value is a string with numbers separated by
    spaces. These can be converted to a numpy array using
    np.fromstring(...).
    An OrderedDict is used to preserve the order of word
    frequency.
    """
    embeddings = OrderedDict()
    with open("i_data/" + emb_fname) as file:
        for line in file:
            idx = line.find(" ")
            word = line[:idx]
            embedding = line[idx+1:].strip()
            embeddings[word] = embedding
    return embeddings