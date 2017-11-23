from collections import Counter
import numpy as np

# Read all words available in the word embeddings file
embeddings = {}
with open("../i_data/glove42B300d.txt") as file:
    for line in file:
        idx = line.find(" ")
        word = line[:idx]
        embedding = line[idx+1:-1]
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
                if word in embeddings:
                    normalized += " " + word
                else:
                    normalized += " |UNK|"
            normalized += " \n"
    output = open(outfname, "w")
    output.write(normalized)
    output.close()

def separate_words(filename, outfname):
    result = ""
    with open(filename) as file:
        for line in file:
            for word in line.split():
                if word not in embeddings:
                    # Check if compound word
                    comp_parts = word.split("-")
                    if len(comp_parts) > 0:
                        # Check if we have word embeddings for all parts
                        all_parts = True
                        for part in comp_parts:
                            all_parts = all_parts and part in embeddings
                        if all_parts:
                            # Replace - with a space
                            word = word.replace("-", " ")

                result += " " + word
            result += "\n"
    output = open(outfname, "w")
    output.write(result)
    output.close()


