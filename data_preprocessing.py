from collections import OrderedDict

def read_embeddings(emb_fname):
    """ Reads """
    embeddings = OrderedDict()
    with open("i_data/" + emb_fname) as file:
        for line in file:
            idx = line.find(" ")
            word = line[:idx]
            embedding = line[idx+1:].strip()
            embeddings[word] = embedding
    return embeddings