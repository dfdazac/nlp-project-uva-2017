from collections import Counter, OrderedDict
import numpy as np

FILES_PATH = "../i_data/"

def missing_embeddings(filename, embeddings):
    missing_counter = Counter()

    with open(filename) as file:
        for line in file:
            words = line.split()
            for word in words:
                if word not in embeddings:
                    missing_counter[word] += 1
    print(len(missing_counter), "embeddings missing in", filename)
    return missing_counter

def normalize_file(filename, outfname, embeddings):
    normalized = ""
    with open(filename) as file:
        for line in file:
            for word in line.split():
                # If embedding is available, don't modify anything
                if word in embeddings:
                    normalized += " " + word                
                else:
                    # If embedding not available, check first
                    # if it's a compound word
                    comp_parts = word.split("-")
                    if len(comp_parts) > 1:
                        # Check if there are embeddings for all parts
                        all_parts = True
                        for part in comp_parts:
                            all_parts = all_parts and part in embeddings
                        if all_parts:
                            # Create new embedding for the word 
                            # as the average of the embeddings of each part
                            part_embedding = np.fromstring(embeddings[comp_parts[0]], sep=" ")
                            for i in range(1, len(comp_parts)):
                                part_embedding += np.fromstring(embeddings[comp_parts[i]], sep=" ")
                            part_embedding /= len(comp_parts)
                            # Convert into properly formatted string and store back in embeddings
                            new_embedding = ""
                            for i in range(len(part_embedding)-1):
                                new_embedding += "{:.5f} ".format(part_embedding[i])
                            embeddings[word] = new_embedding + "{:.5f}".format(part_embedding[-1])
                            # Add the word as it is, because we now have an embedding for it
                            normalized += " " + word
                    # If not a compound word add |UNK| entry
                    else:
                        normalized += " |UNK|"
            normalized += " \n"
    with open(outfname, "w") as file:
        file.write(normalized)

# Read base embeddings
emb_fname = "glove42B300d.txt"
print("Loading embeddings...")
embeddings = OrderedDict()
with open(FILES_PATH + emb_fname) as file:
    for line in file:
        idx = line.find(" ")
        word = line[:idx]
        embedding = line[idx+1:].strip()
        embeddings[word] = embedding

corpus_files = ["train.txt", "valid.txt", "test.txt"]
# Normalize all files and update embeddings for compound words
for fname in corpus_files:
    missing_embeddings(FILES_PATH + fname, embeddings)
    print("Normalizing", fname)
    normalize_file(FILES_PATH + fname, FILES_PATH + "n_" + fname, embeddings)
    for m in missing_embeddings(FILES_PATH + "n_" + fname, embeddings).most_common():
        print(m)

print("Reducing embeddings...")
vocab_counter = Counter()
for fname in corpus_files:
    with open(FILES_PATH + fname) as file:
        for line in file:
            vocab_counter.update(line.split())
unused = []
reduced_embs = OrderedDict()
for word in embeddings:
    if word in vocab_counter:
        reduced_embs[word] = embeddings[word]
    else:
        unused.append(word)
#reduced_embs = OrderedDict({word: embeddings[word] for word in embeddings if word in vocab_counter})
# Attempot to complete 10k words
for un_word in unused:
    if len(reduced_embs) >= 10000:
        break
    else:
        reduced_embs[un_word] = embeddings[word]

print("Embeddings reduced from", len(embeddings), "to", len(reduced_embs))

# Finally, take the average of all embeddings
# and use that for the word embedding of |UNK|
print("Averaging word embeddings for |UNK|...")
unk_embedding = np.zeros(np.fromstring(reduced_embs[next(iter(reduced_embs))], sep=" ").shape)
for k in reduced_embs:
    try:
        unk_embedding += np.fromstring(reduced_embs[k], sep=" ")
    except:
        print(k, reduced_embs[k])
        break
unk_embedding /= len(reduced_embs)
new_embedding = ""
for i in range(len(unk_embedding)-1):
    new_embedding += "{:.5f} ".format(unk_embedding[i])
new_embedding += "{:.5f}".format(unk_embedding[-1])
reduced_embs["|UNK|"] = new_embedding

for fname in ["n_train.txt", "n_valid.txt", "n_test.txt"]:
    missing_embeddings(FILES_PATH + fname, reduced_embs)

# Save final word embeddings
print("Saving reduced word embeddings...")
with open(FILES_PATH + "n_" + emb_fname, "w") as file:
    for k in reduced_embs:
        file.write(k + " " + reduced_embs[k] + "\n")
print("Reduced embeddings saved at " + FILES_PATH + "n_" + emb_fname)
