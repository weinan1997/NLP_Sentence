
# coding: utf-8

# In[3]:


import numpy as np
from collections import defaultdict
import sys, re
import pandas as pd
import gensim
import random
import torch


# In[4]:


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


# In[5]:


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


# In[6]:


# def load_bin_vec(fname, vocab):
#     """
#     Loads 300x1 word vecs from Google (Mikolov) word2vec
#     """
#     word_vecs = {}
#     with open(fname, "rb") as f:
#         header = f.readline()
#         vocab_size, layer1_size = map(int, header.split())
#         binary_len = np.dtype('float32').itemsize * layer1_size
#         for line in range(vocab_size):
#             word = []
#             while True:
#                 ch = f.read(1)
#                 if ch == ' ':
#                     word = ''.join(word)
#                     break
#                 if ch != '\n':
#                     word.append(ch)   
#             if word in vocab:
#                 word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
#             else:
#                 f.read(binary_len)
#     return word_vecs

def load_bin_vec(fname, vocab):
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
    for word in vocab:
        if word in model:
            word_vecs[word] = model[word]
    return word_vecs


# In[7]:


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  


# In[8]:


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_review = data_folder[0]
    neg_review = data_folder[1]
    vocab = defaultdict(float)
    for review in pos_review:   
        rev = []
        rev.append(review.strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum  = {"y":1, 
                  "text": orig_rev,                             
                  "num_words": len(orig_rev.split()),
                  "split": np.random.randint(0,cv)}
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
            vocab[word] += 1
        datum  = {"y":0, 
                  "text": orig_rev,                             
                  "num_words": len(orig_rev.split()),
                  "split": np.random.randint(0,cv)}
        revs.append(datum)
    random.shuffle(revs)
    return revs, vocab


# In[9]:


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

def make_idx_data(revs, word_idx_map, max_l=100, k=300):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, dev, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l)
        sent.append(rev["y"])
        if rev["split"]==8:            
            dev.append(sent)        
        elif rev["split"]>=9:  
            test.append(sent)   
        else:
            train.append(sent)
    train = np.array(train,dtype="int")
    dev = np.array(dev,dtype="int")
    test = np.array(test,dtype="int")
    return [train, dev, test]     

# In[21]:


def data_process(pos_file, neg_file, max_l=100):  
    pos_reviews = review_extract(pos_file)
    neg_reviews = review_extract(neg_file)
    data_folder = [pos_reviews, neg_reviews]
    print("loading data...")  
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...",)
    w2v = load_bin_vec("GoogleNews-vectors-negative300.bin", vocab)
    print("finish loading")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map= get_W(w2v)
    
    print("create train, dev, and test sets")
    processed_data = []
    data_set = make_idx_data(revs, word_idx_map, max_l)
    for data in data_set:
        X = []
        Y = []
        for index_array in data:
            x = []
            for index in index_array[0:-1]:
                x.append(W[index])
            x = np.matrix(x)
            X.append(x)
            Y.append(index_array[-1])
        processed_data.append([X, Y])
    print("finish creating")
    return processed_data

def save_data(data_set, max_l=100):
    data = data_process("sorted_data/"+data_set+"/positive.review", "sorted_data/"+data_set+"/negative.review")
    torch.save(data, data_set + "_processed.wordvec")
    print("data saved")

def load_data(file_name):
    data = torch.load(file_name)
    print("data loaded")
    return data



