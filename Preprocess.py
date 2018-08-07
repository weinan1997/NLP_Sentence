import numpy as np
from collections import defaultdict
import sys, re
import pandas as pd
import gensim
import random
import torch
import os


#Data Preprocess
def review_extract(filename):
    file = open(filename, "r+", encoding="ISO-8859-1")
    text = file.read()
    file.close()
    pattern = re.compile(r'<review_text>.*?</review_text>', re.DOTALL)
    results = pattern.findall(text)
    clean_reviews = []
    for result in results:
        temp1 = result.replace("<review_text>", "")
        temp2 = temp1.replace("</review_text>", "")
        clean_reviews.append(temp2)
    return clean_reviews


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()


def load_bin_vec(fname, vocab):
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    for word in vocab:
        if word in model:
            word_vecs[word] = model[word]
    return word_vecs


def add_unknown_words(word_vecs, vocab, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  


def build_data(data_folder, domain, cv=5, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_review = data_folder[0]
    neg_review = data_folder[1]
    vocab = set()
    for review in pos_review:   
        rev = []
        rev.append(review.strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab.add(word)
        datum  = {"y":1, 
                  "domain":domain,
                  "text": orig_rev,                             
                  "num_words": len(orig_rev.split()),
                  "split": np.random.randint(0, cv)}
        revs.append(datum)
    for review in neg_review:   
        rev = []
        rev.append(review.strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab.add(word)
        datum  = {"y":0, 
                  "domain":domain,
                  "text": orig_rev,                             
                  "num_words": len(orig_rev.split()),
                  "split": np.random.randint(0, cv)}
        revs.append(datum)
    random.shuffle(revs)
    return revs, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map   


def data_process(max_l):  
    max_remain = 0
    data_file = ["books", "dvd", "electronics", "kitchen"]
    revs_dict, vocab_list = {}, []
    domain = 0
    for file_name in data_file:
        path = "../../../data1/weinan/"
        if os.path.exists(path):
            pos_file = path+"sorted_data/"+file_name+"/positive.review"
            neg_file = path+"sorted_data/"+file_name+"/negative.review"
        else:
            pos_file = "sorted_data/"+file_name+"/positive.review"
            neg_file = "sorted_data/"+file_name+"/negative.review"
        pos_reviews = review_extract(pos_file)
        neg_reviews = review_extract(neg_file)
        data_folder = [pos_reviews, neg_reviews]
        print("loading data of {}...".format(file_name))  
        revs, vocab = build_data(data_folder, domain, clean_string=True)
        domain = domain + 1
        revs_dict[file_name] = revs
        vocab_list.append(vocab)
        max_length = np.max(pd.DataFrame(revs)["num_words"])
        print("data loaded!")
        print("number of sentences: " + str(len(revs)))
        print("vocab size: " + str(len(vocab)))
        print("max sentence length: " + str(max_length))
        remain_l = int(np.percentile(pd.DataFrame(revs)["num_words"], max_l))
        print("remain sentence length: " + str(remain_l) + "\n")
        max_remain = max(max_remain, remain_l)
    print("max remain sentence length: " + str(max_remain) + "\n")
    vocab = set()
    for v in vocab_list:
        vocab.update(v)
    print("loading word2vec vectors...",)
    path = "../../../data1/GoogleNews-vectors-negative300.bin"
    if not os.path.exists(path):
        path = "GoogleNews-vectors-negative300.bin"
    w2v = load_bin_vec(path, vocab)
    print("finish loading")
    print("num words already in word2vec: " + str(len(w2v)) + "\n")
    add_unknown_words(w2v, vocab)
    W, word_idx_map= get_W(w2v)

    torch.save([revs_dict, W, word_idx_map], "revs_W_map.matrix")

    return max_remain
 
 