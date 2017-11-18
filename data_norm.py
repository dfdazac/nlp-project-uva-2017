from collections import Counter, OrderedDict
import numpy as np

# Read base embeddings
emb_fname = "glove42B300d.txt"
print("Loading embeddings...")
embeddings = OrderedDict()
with open("i_data/" + emb_fname) as file:
    for line in file:
        idx = line.find(" ")
        word = line[:idx]
        embedding = line[idx+1:].strip()
        embeddings[word] = embedding

def missing_embeddings(filename):
    missing_counter = Counter()

    with open(filename) as file:
        for line in file:
            words = line.split()
            for word in words:
                if word not in embeddings:
                    missing_counter[word] += 1
    print(len(missing_counter), "embeddings missing in", filename)
    return missing_counter

def normalize_file(filename, outfname):
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

# Normalize all files and update embeddings for compound words
for fname in ["train.txt", "valid.txt", "test.txt"]:
    missing_embeddings("i_data/" + fname)
    print("Normalizing file...")
    normalize_file("i_data/" + fname, "i_data/n_" + fname)
    for m in missing_embeddings("i_data/n_" + fname).most_common():
        print(m)

# Finally, take the average of all embeddings
# and use that for the word embedding of |UNK|
print("Averaging word embeddings for |UNK|...")
unk_embedding = np.zeros(np.fromstring(embeddings[next(iter(embeddings))], sep=" ").shape)
for k in embeddings:
    try:
        unk_embedding += np.fromstring(embeddings[k], sep=" ")
    except:
        print(k, embeddings[k])
        break
unk_embedding /= len(embeddings)
new_embedding = ""
for i in range(len(unk_embedding)-1):
    new_embedding += "{:.5f} ".format(unk_embedding[i])
new_embedding += "{:.5f}".format(unk_embedding[-1])
embeddings["|UNK|"] = new_embedding

for fname in ["n_train.txt", "n_valid.txt", "n_test.txt"]:
    missing_embeddings("i_data/" + fname)

# Save final word embeddings
print("Saving word embeddings...")
with open("i_data/n_" + emb_fname, "w") as file:
    for k in embeddings:
        file.write(k + " " + embeddings[k] + "\n")
print("New embeddings saved at i_data/n_" + emb_fname)
