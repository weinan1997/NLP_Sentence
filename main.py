import os
import torch
import CNN_model
import LSTM_model
import Attention_model
import Train
import Preprocess
import sys
import numpy as np

args = {"model":"attention", "vec_len":300, "max_l":90, "filter_num":3, "kernel_sizes":[3,4,5], "kernel_num":100, "dropout":0.5, "batch_size":50, "epoch_num":30, "early_stop":5, "data_set":"dvd", "eval":True, "lstm_dim":300, "layer_num":1, "attention_dim":100}
args["data_set"] = sys.argv[1]

data = []
if args["data_set"] == "all":
    set1 = np.array(torch.load("books.wordvec"))
    set2 = np.array(torch.load("dvd.wordvec"))
    set3 = np.array(torch.load("electronics.wordvec"))
    set4 = np.array(torch.load("kitchen.wordvec"))
    d_set = [set1, set2, set3, set4]
    for i in range(0, 3):
        X, Y = [], []
        for j in range(0, len(d_set)):
            X = X + d_set[j][i][0]
            Y = Y + d_set[j][i][1]
        data.append([X, Y])
else:
    if not os.path.exists(args["data_set"] + ".wordvec"):
        Preprocess.save_data(args["max_l"])
    data = Preprocess.load_data(args["data_set"] + ".wordvec")

if args["eval"] == True:
    model = torch.load("all_"+args["model"]+".model")
    Train.eval(data[2], model, args)
    exit()

if args["model"] == "cnn":
    model = CNN_model.CNN_Sentence(args)
elif args["model"] == "lstm":
    model = LSTM_model.LSTM_Sentence(args)
elif args["model"] == "attention":
    model = Attention_model.Attention_Sentence(args)
Train.train(data[0], data[1], model, args)
print('\nTest set result:\n')
Train.eval(data[2], model, args)