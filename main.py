import os
import torch
import CNN_model
import GRU_model
import GRU_Attention_model
import CNN_Attention_model
import SWEM_hier_model
import CNN_Att_Pool_model
import Train
import Preprocess
import sys
import numpy as np
import argparse
import random

domain_set = ["books", "dvd", "electronics", "kitchen"]

def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment Analysis")
    parser.add_argument('-m', "--model", default="cnn", help="available models: cnn, gru, gru_attention, cnn_att_pool")
    parser.add_argument("--max_l", default=90, type=int, help="percentage of sentences' lengths")
    parser.add_argument("--kernel_sizes", default=[3,4,5], type=list, help="kernel sizes for convolution")
    parser.add_argument("--kernel_num", default=100, type=int, help="number of output filters")
    parser.add_argument("--dropout", default=0.5, type=float, help="dropout")
    parser.add_argument('-b', "--batch_size", default=50, type=int, help="batch size")
    parser.add_argument('-e', "--epoch_num", default=30, type=int,help="Maximum epoch number")
    parser.add_argument("--early_stop", default=5, type=int, help="early stop number")
    parser.add_argument('-d', "--data_set", default="books", help="available data set: books, dvd, electronics, kitchen, all")
    parser.add_argument("--eval", default=False, type=bool, help="train or evaluation")
    parser.add_argument("--hidden_size", default=100, type=int, help="hidden size for LSTM")
    parser.add_argument('-a', "--attention_dim", default=100, type=int, help="attention size")
    parser.add_argument('-g', "--gpu", default=0, type=int,help="GPU number")
    parser.add_argument('-s', "--seed", default=1, type=int, help="set random seed")
    parser.add_argument('-c', "--cross_validation", default=9, type=int, help="set the data set as test set")
    parser.add_argument("--run_cv", default=False, type=bool, help="run through all cross validation test")
    parser.add_argument("--test_all", default=False, type=bool, help="train model on the whole set and test in each domain")
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
        "seed": options.seed,
        "cross_validation": options.cross_validation,
        "run_cv": options.run_cv,
        "test_all": options.test_all,
        "vec_len": 300,
        "layer_num": 1,
        "remain_l": 426
    }
    return args


def partition_data(args):
    data = []
    data_array = []
    if args["data_set"] == "all":
        set1 = np.array(torch.load("books.wordvec"))
        set2 = np.array(torch.load("dvd.wordvec"))
        set3 = np.array(torch.load("electronics.wordvec"))
        set4 = np.array(torch.load("kitchen.wordvec"))
        d_set = [set1, set2, set3, set4]
        for i in range(0, 10):
            X, Y = [], []
            for j in range(0, len(d_set)):
                X = X + d_set[j][i][0]
                Y = Y + d_set[j][i][1]
            data_array.append([X, Y])
    else:
        if not os.path.exists(args["data_set"] + ".wordvec"):
            args["remain_l"] = Preprocess.save_data(args["max_l"])
        data_array = Preprocess.load_data(args["data_set"] + ".wordvec")

    print("partitioning data set...")
    test_index = args["cross_validation"]
    test_set = data_array[test_index]
    dev_index = random.randint(0, 9)
    while dev_index == args["cross_validation"]:
        dev_index = random.randint(0, 9)
    dev_set = data_array[dev_index]
    train_set = []
    for i in range(len(data_array)):
        if i == args["cross_validation"] or i == dev_index:
            continue
        train_set.append(data_array[i])
    X, Y = [], []
    for i in range(len(train_set)):
        X = X + train_set[i][0]
        Y = Y + train_set[i][1]
    train_set = [X, Y]
    data = [train_set, dev_set, test_set]
    print("finish partitioning!")
    return data
        
def find_model(args):
    if args["model"] == "cnn":
        model = CNN_model.CNN_Sentence(args)
    elif args["model"] == "gru":
        model = GRU_model.GRU_Sentence(args)
    elif args["model"] == "gru_attention":
        model = GRU_Attention_model.GRU_Attention_Sentence(args)
    elif args["model"] == "cnn_attention":
        model = CNN_Attention_model.CNN_Attention_Sentence(args)
    elif args["model"] == "SWEM_hier":
        model = SWEM_hier_model.SWEM_hier_Sentence(args)
    elif args["model"] == "cnn_att_pool":
        model = CNN_Att_Pool_model.CNN_Att_Pool_Sentence(args)
    else:
        print("No such model!")
        exit()
    if torch.cuda.is_available():
        model = model.cuda(args["GPU"])
    return model

def main():
    args = parse_args()
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    data = []

    if args["eval"]:
        model = torch.load("all_"+args["model"]+".model")
        data = partition_data(args)
        Train.eval(data[2], model, args)
        exit()

    torch.cuda.manual_seed_all(args["seed"])

    if args["test_all"]:
        args["data_set"] = "all"
        result_list = []
        for i in range(10):
            args["cross_validation"] = i
            data = partition_data(args)
            model = find_model(args)
            Train.train(data[0], data[1], model, args)
            print('\nTest set result:\n')
            all_result = Train.eval(data[2], model, args)
            model = torch.load("all_"+args["model"]+".model")
            result = []
            for domain in domain_set:
                args["data_set"] = domain
                data = partition_data(args)
                result.append(Train.eval(data[2], model, args))
            result.append(all_result)
            result = np.array(result)
            result_list.append(result)
        np.set_printoptions(precision=4)
        result_list = np.array(result_list)
        print(result_list)
        exit()
        


    if args["run_cv"]:
        result_list = []
        for i in range(10):
            args["cross_validation"] = i
            data = partition_data(args)
            model = find_model(args)
            Train.train(data[0], data[1], model, args)
            print('\nTest set result:\n')
            result_list.append(Train.eval(data[2], model, args))
        result = np.array(result_list)
        np.set_printoptions(precision=4)
        print(result)
        print('Average: {:.4f}%,    Standard Deviation: {:.4f}%'.format(result.mean(), result.std()))
    else:
        data = partition_data(args)
        Train.train(data[0], data[1], model, args)
        print('\nTest set result:\n')
        Train.eval(data[2], model, args)

if __name__ == "__main__":
    main()