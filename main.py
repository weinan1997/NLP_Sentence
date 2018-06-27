import os
import torch
import CNN_model
import LSTM_model
import GRU_Attention_model
import CNN_Attention_model
import SWEM_hier_model
import CNN_Att_Pool_model
import Train
import Preprocess
import sys
import numpy as np

args = {"model":"cnn_att_pool", "vec_len":300, "max_l":90, "remain_l":426, "filter_num":3, "kernel_sizes":[3,4,5], "kernel_num":100, "dropout":0.5, "batch_size":50, "epoch_num":20, "early_stop":5, "data_set":"dvd", "eval":False, "lstm_dim":300, "layer_num":1, "attention_dim":100, "GPU":0}
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
        args["remain_l"] = Preprocess.save_data(args["max_l"])
    data = Preprocess.load_data(args["data_set"] + ".wordvec")

if args["eval"] == True:
    model = torch.load("all_"+args["model"]+".model")
    Train.eval(data[2], model, args)
    exit()

if args["model"] == "cnn":
    model = CNN_model.CNN_Sentence(args)
elif args["model"] == "lstm":
    model = LSTM_model.LSTM_Sentence(args)
elif args["model"] == "gru_attention":
    model = GRU_Attention_model.GRU_Attention_Sentence(args)
elif args["model"] == "cnn_attention":
    model = CNN_Attention_model.CNN_Attention_Sentence(args)
elif args["model"] == "SWEM_hier":
    model = SWEM_hier_model.SWEM_hier_Sentence(args)
elif args["model"] == "cnn_att_pool":
    model = CNN_Att_Pool_model.CNN_Att_Pool_Sentence(args)
if torch.cuda.is_available():
    model = model.cuda(args["GPU"])
Train.train(data[0], data[1], model, args)
print('\nTest set result:\n')
Train.eval(data[2], model, args)
