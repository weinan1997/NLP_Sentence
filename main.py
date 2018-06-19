import os
import torch
import CNN_model
import Train
import Preprocess
import sys
import numpy as np

args = {"vec_len":300, "max_l":100, "filter_num":3, "kernel_sizes":[3,4,5], "kernel_num":100, "dropout":0.5, "batch_size":50, "epoch_num":20, "data_set":"dvd", "eval":True}
args["data_set"] = sys.argv[1]

if args["eval"] == True:
    model = torch.load("all_cnn.model")
    data = Preprocess.load_data(args["data_set"] + "_processed.wordvec")
    Train.eval(data[2], model, args)
    exit()

data = []
if args["data_set"] == "all":
    set1 = np.array(Preprocess.load_data("books_processed.wordvec"))
    set2 = np.array(Preprocess.load_data("dvd_processed.wordvec"))
    set3 = np.array(Preprocess.load_data("electronics_processed.wordvec"))
    set4 = np.array(Preprocess.load_data("kitchen_processed.wordvec"))
    d_set = [set1, set2, set3, set4]
    for i in range(0, 3):
        X, Y = [], []
        for j in range(0, len(d_set)):
            X = X + d_set[j][i][0]
            Y = Y + d_set[j][i][1]
        data.append([X, Y])
else:
    if not os.path.exists(args["data_set"] + "_processed.wordvec"):
        Preprocess.save_data(args["data_set"], args["max_l"])
    else:
        data = Preprocess.load_data(args["data_set"] + "_processed.wordvec")
model = CNN_model.CNN_Sentence(args)
Train.train(data[0], data[1], model, args)
print('\nTest set result:\n')
Train.eval(data[2], model, args)