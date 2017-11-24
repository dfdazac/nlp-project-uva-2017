from subprocess import run
from os import listdir
import kenlm
import lm_eval as lme
import numpy as np
import matplotlib.pyplot as plt

# Training and validation corpora
train_fname = "train.txt.UNK"
valid_fname = "valid.txt.UNK"
# The relative path containing the training and validation corpora
data_path = "../data/"
# The path where arpa and binary files will be saved
output_path = "../i_data/"
# Path to KenLM binaries
klmbin_path = "/home/daniel/kenlm/build/bin/"


computed_files = listdir(output_path)

orders = np.array([2, 3, 4, 5, 6])
train_perplexities = np.zeros(len(orders))
valid_perplexities = np.zeros(len(orders))

# Compute arpa and binary files for desired orders
for i, n_order in enumerate(orders):
    order = str(n_order)
    arpa_fname = order + "_" + train_fname + ".arpa"
    bin_fname = order + "_" + train_fname + ".bin"

    # If such files exist, don't recompute them
    if arpa_fname not in computed_files:
        run(" ".join([klmbin_path + "lmplz -o", order, "<", data_path + train_fname, ">", output_path + arpa_fname]), shell=True)
    if bin_fname not in computed_files:
        run(" ".join([klmbin_path + "build_binary", output_path + arpa_fname, output_path + bin_fname]), shell=True)

    # Evaluate model on training and validation sets
    model = kenlm.LanguageModel(output_path + bin_fname)
    train_perplexities[i] = lme.perplexity(data_path + train_fname, model.score, base=10)
    valid_perplexities[i] = lme.perplexity(data_path + valid_fname, model.score, base=10)

# Show bar plots with perplexities
bar_width = 0.35
index = np.arange(len(orders))
plt.grid(zorder=0)
plt.bar(index, train_perplexities, bar_width, label="Training", zorder=3)
plt.bar(index + bar_width, valid_perplexities, bar_width, label="Validation", zorder=3)
plt.xticks(index + bar_width/2, orders)
plt.xlabel("N-gram order")
plt.title("Perplexity")
plt.legend()
plt.show()
