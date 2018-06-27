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
import argparse

parser = argparse.ArgumentParser(description="Sentiment Analysis")
parser.add_argument('-m', "--model", default="cnn", help="available models: cnn, lstm, gru_attention, cnn_att_pool")
parser.add_argument("--max_l", default=90, help="percentage of sentences' lengths")
parser.add_argument("--kernel_sizes", default=[3,4,5], help="kernel sizes for convolution")
parser.add_argument("--kernel_num", default=100, help="number of output filters")
parser.add_argument("--dropout", default=0.5, help="dropout")
parser.add_argument('-b', "--batch_size", default=50, help="batch size")
parser.add_argument('-e', "--epoch_num", default=20, help="Maximum epoch number")
parser.add_argument("--early_stop", default=3, help="early stop number")
parser.add_argument('-d', "--data_set", default="books", help="available data set: books, dvd, electronics, kitchen, all")
parser.add_argument("--eval", default=False, type=bool, help="train or evaluation")
parser.add_argument("--hidden_size", default=100, help="hidden size for LSTM")
parser.add_argument('-a', "--attention_dim", default=100, help="attention size")
parser.add_argument('-g', "--gpu", default=0, help="GPU number")
options = parser.parse_args()
args = {
    "model": options.model,
    "max_l": options.max_l,
    "kernel_sizes": options.kernel_sizes,
    "kernel_num": options.kernel_num,
    "dropout": options.dropout,
    "batch_size": options.batch_size,
    "epoch_num": options.epoch_num,
    "early_stop": options.early_stop,
    "data_set": options.data_set,
    "eval": options.eval,
    "lstm_dim": options.hidden_size,
    "attention_dim": options.attention_dim,
    "GPU": options.gpu,
    "vec_len": 300,
    "layer_num": 1,
    "remain_l": 426
}

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
