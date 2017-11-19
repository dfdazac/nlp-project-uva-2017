from subprocess import run
from os import listdir
import kenlm
import lm_eval as lme

klmbin_path = "home/daniel/kenlm/build/bin/"
data_path = "data/"
output_path = "i_data/"
train_fname = "train.txt"
valid_fname = "valid.txt"

computed_files = listdir(output_path)

orders = [2]

# Compute arpa and binary files for desired ordersTrain model of given order
for n_order in orders:    
    order = str(n_order)
    arpa_fname = order + "_" + train_fname + ".arpa"
    bin_fname = order + "_" + train_fname + ".bin"

    # If such files exist, don't recompute them
    if arpa_fname not in computed_files:
        run(" ".join([klmbin_path + "lmplz -o", order, "<", data_path + train_fname, ">", output_path + arpa_fname]), shell=True)
    if bin_fname not in computed_files:
        run(" ".join([klmbin_path + "build_binary", output_path + arpa_fname, output_path + bin_fname]), shell=True)

    # Evaluate model
    model = kenlm.LanguageModel(output_path + bin_fname)
    print(lme.perplexity(data_path + train_fname, model.score))
