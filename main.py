import os
import torch
import CNN_model
import GRU_model
import GRU_Attention_model
import CNN_Attention_model
import CNN_Att_Pool_model
import DAM_model
import GS_model
import Train
import Preprocess
import sys
import numpy as np
import argparse
import random
import pickle

domain_set = ["books", "dvd", "electronics", "kitchen"]
model_set = ["cnn", "cnn_attention", "gru", "gru_attention", "dam"]

def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment Analysis")
    parser.add_argument('-m', "--model", default="cnn", help="available models: cnn, gru, gru_attention, cnn_att_pool, gs")
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
    parser.add_argument('-c', "--cross_validation", default=0, type=int, help="set the data set as test set")
    parser.add_argument("--run_cv", default=False, type=bool, help="run through all cross validation test")
    parser.add_argument("--test_all", default=False, type=bool, help="train model on the whole set and test in each domain")
    parser.add_argument('-r', "--regular", default=0.03, type=float, help="set the regular coefficient for DAM")
    parser.add_argument("--lambda1", default=0.04, type=float, help="lambda1 for GS model")
    parser.add_argument("--lambda2", default=0.01, type=float, help="lambda2 for GS model")
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
        "regular": options.regular,
        "lambda1": options.lambda1,
        "lambda2": options.lambda2,
        "vec_len": 300,
        "layer_num": 1,
        "remain_l": 426
    }
    return args


# def partition_data(args):
#     data = []
#     data_array = []
#     path = "../../../data1/weinan/"
#     if args["data_set"] == "all":
#         if os.path.exists(path):
#             set1 = np.array(torch.load(path+"books.wordvec"))
#             set2 = np.array(torch.load(path+"dvd.wordvec"))
#             set3 = np.array(torch.load(path+"electronics.wordvec"))
#             set4 = np.array(torch.load(path+"kitchen.wordvec"))
#         else:
#             set1 = np.array(torch.load("books.wordvec"))
#             set2 = np.array(torch.load("dvd.wordvec"))
#             set3 = np.array(torch.load("electronics.wordvec"))
#             set4 = np.array(torch.load("kitchen.wordvec"))
#         d_set = [set1, set2, set3, set4]
#         for i in range(0, 10):
#             X, Y, Z= [], [], []
#             for j in range(0, len(d_set)):
#                 X = X + d_set[j][i][0]
#                 Y = Y + d_set[j][i][1]
#                 Z = Z + d_set[j][i][2]
#             data_array.append([X, Y, Z])
#     else:
#         if not os.path.exists(path):
#             if not os.path.exists(args["data_set"] + ".wordvec"):
#                 args["remain_l"] = Preprocess.save_data(args["max_l"])
#             data_array = torch.load(args["data_set"] + ".wordvec")
#         else:
#             if not os.path.exists(path + args["data_set"] + ".wordvec"):
#                 args["remain_l"] = Preprocess.save_data(args["max_l"])
#             data_array = torch.load(path + args["data_set"] + ".wordvec")

#     print("partitioning data set...")
#     test_index = args["cross_validation"]
#     test_set = data_array[test_index]
#     dev_index = random.randint(0, 9)
#     while dev_index == args["cross_validation"]:
#         dev_index = random.randint(0, 9)
#     dev_set = data_array[dev_index]
#     train_set = []
#     for i in range(len(data_array)):
#         if i == args["cross_validation"] or i == dev_index:
#             continue
#         train_set.append(data_array[i])
#     X, Y, Z = [], [], []
#     for i in range(len(train_set)):
#         X = X + train_set[i][0]
#         Y = Y + train_set[i][1]
#         Z = Z + train_set[i][2]
#     train_set = [X, Y, Z]
#     data = [train_set, dev_set, test_set]
#     print("finish partitioning!")
#     return data
        

def get_idx_from_sent(sent, word_idx_map, max_l, k=300):
    """
    Transforms sentence into a list of indices. Pad with zeroes. Remove words that exceed max_l.
    """
    x = []
    words = sent.split()
    if len(words) > max_l:
        words = words[:max_l]
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:
        x.append(0)
    return x

def make_idx_data(revs, word_idx_map, max_l, cv, k=300):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, dev, test = [], [], []
    temp = []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l)
        sent.append(rev["y"])
        sent.append(rev["domain"])
        if rev["split"] == cv:
            test.append(sent)
        else:
            temp.append(sent)
    dev_point = int(len(temp) * 0.9)
    train = np.array(temp[:dev_point], dtype="int")
    dev = np.array(temp[dev_point:], dtype="int")
    test = np.array(test, dtype="int")
    return train, dev, test


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
    elif args["model"] == "dam":
        model = DAM_model.DAM_Sentence(args)
    elif args["model"] == "gs":
        model = GS_model.GS_Sentence(args)
    else:
        print("No such model!")
        exit()
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["GPU"])
        model = model.cuda(args["GPU"])
    return model

def main():
    args = parse_args()
    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    if not os.path.exists("revs_W_map.matrix"):
        args["remain_l"] = Preprocess.data_process(args["max_l"])
    revs_dict, args["W"], word_idx_map = torch.load("revs_W_map.matrix")
    
    data = []

    if args["eval"]:
        model = torch.load("all_"+args["model"]+".model")
        revs = revs_dict[args["data_set"]]
        data = make_idx_data(revs, word_idx_map, args["remain_l"], args["cross_validation"])
        Train.eval(data[2], model, args)
        exit()

    torch.cuda.manual_seed_all(args["seed"])

    if args["test_all"] and args["run_cv"]:
        args["data_set"] = "all"
        revs = []
        for domain in domain_set:
            revs += revs_dict[domain]
        random.shuffle(revs)
        result_list = []
        for i in range(5):
            args["cross_validation"] = i
            data = make_idx_data(revs, word_idx_map, args["remain_l"], args["cross_validation"])
            model = find_model(args)
            Train.train(data[0], data[1], model, args)
            print('\nTest set result:\n')
            if args["model"] == "gs":
                model = torch.load("gs"+"_hd_"+str(args["attention_dim"])+"_l1_"+str(args["lambda1"])+"_l2_"+str(args["lambda2"])+".model")
            else:
                model = torch.load("all_"+args["model"]+".model")
            result = []
            all_result = Train.eval(data[2], model, args)
            for domain in domain_set:
                sub_revs = revs_dict[domain]
                data = make_idx_data(sub_revs, word_idx_map, args["remain_l"], args["cross_validation"])
                result.append(Train.eval(data[2], model, args))
            result.append(all_result)
            result = np.array(result)
            result_list.append(result)
        np.set_printoptions(precision=4)
        result_list = np.array(result_list)
        print(result_list)
        exit()
    
    if args["test_all"]:
        args["data_set"] = "all"
        revs = []
        for domain in domain_set:
            revs += revs_dict[domain]
        random.shuffle(revs)
        data = make_idx_data(revs, word_idx_map, args["remain_l"], args["cross_validation"])
        model = find_model(args)
        Train.train(data[0], data[1], model, args)
        print('\nTest set result:\n')
        if args["model"] == "gs":
            model = torch.load("gs"+"_hd_"+str(args["attention_dim"])+"_l1_"+str(args["lambda1"])+"_l2_"+str(args["lambda2"])+".model")
        else:
            model = torch.load("all_"+args["model"]+".model")
        result = []
        all_result = Train.eval(data[2], model, args)
        for domain in domain_set:
            sub_revs = revs_dict[domain]
            data = make_idx_data(sub_revs, word_idx_map, args["remain_l"], args["cross_validation"])
            result.append(Train.eval(data[2], model, args))
        result.append(all_result)
        result = np.array(result)
        np.set_printoptions(precision=4)
        print(result)
        exit()
        


    if args["run_cv"]:
        result_list = []
        for i in range(5):
            args["cross_validation"] = i
            revs = revs_dict[args["data_set"]]
            data = make_idx_data(revs, word_idx_map, args["remain_l"], args["cross_validation"])
            model = find_model(args)
            Train.train(data[0], data[1], model, args)
            print('\nTest set result:\n')
            result_list.append(Train.eval(data[2], model, args))
        result = np.array(result_list)
        result = result.reshape(10, 1)
        np.set_printoptions(precision=4)
        print(result)
        print('Average: {:.4f}%,    Standard Deviation: {:.4f}%'.format(result.mean(), result.std()))
    else:
        revs = revs_dict[args["data_set"]]
        data = make_idx_data(revs, word_idx_map, args["remain_l"], args["cross_validation"])
        model = find_model(args)
        Train.train(data[0], data[1], model, args)
        print('\nTest set result:\n')
        Train.eval(data[2], model, args)

if __name__ == "__main__":
    main()