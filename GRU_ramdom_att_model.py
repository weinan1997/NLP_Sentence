import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU_random_att_Sentence(nn.Module):

    def __init__(self, args):
        super(GRU_random_att_Sentence, self).__init__()
        self.args = args
        
        self.hidden_dim = args["attention_dim"]
        self.vec_len = args["vec_len"]
        self.batch_size = args["batch_size"]
        self.word2vec = args["W"]
        self.dp = args["dropout"]
        self.domain_num = args["domain_num"]
        self.seq_len = args["remain_l"]
        self.gpu = args["GPU"]
        self.att_len = 500

        self.embedding = nn.Embedding(self.word2vec.shape[0], self.vec_len)
        self.embedding.weight.data.copy_(torch.from_numpy(self.word2vec))
        self.gru = nn.GRU(self.vec_len, self.hidden_dim, bidirectional=True)
        self.att_weight = nn.Parameter(torch.FloatTensor(self.att_len, 1))
        torch.nn.init.xavier_normal_(self.att_weight)
        self.att_fc = nn.Linear(2*self.hidden_dim, self.att_len)
        self.dropout = nn.Dropout(self.dp, inplace=True)
        self.fc = nn.Linear(2*self.hidden_dim, 2)

        # self.ew = torch.tensor(self.att_weight.data)

    def forward(self, x):
        # print(torch.norm(self.ew - self.att_weight.data)/torch.norm(self.ew))
        x = self.embedding(x)
        h, _ = self.gru(x.permute(1, 0, 2))
        h = h.permute(1, 0, 2)
        hr = h.reshape(h.shape[0]*h.shape[1], h.shape[2])
        u = F.tanh(self.att_fc(hr))
        u = u.reshape(h.shape[0], h.shape[1], -1)
        att_applied = F.softmax(torch.matmul(u, self.att_weight), dim=1)
        h = h.permute(0, 2, 1)
        output = torch.bmm(h, att_applied)
        output = output.squeeze(2)
        self.dropout(output)
        # print(self.att_weight.data[0, 0:10])
        output = self.fc(output)
        return output

    def calculate_att_weight(self, revs, word_idx_map):
        pword_weight, pword_num, nword_weight, nword_num = {}, {}, {}, {}
        voca = set()
        with torch.no_grad():
            x = []
            for rev in revs:
                idx = self.get_idx_from_sent(rev["text"], word_idx_map, self.seq_len)
                x.append(idx)
            x = torch.tensor(x)
            if torch.cuda.is_available():
                x = x.cuda()
            x = self.embedding(x)
            h, _ = self.gru(x.permute(1, 0, 2))
            h = h.permute(1, 0, 2)
            hr = h.reshape(h.shape[0]*h.shape[1], h.shape[2])
            u = F.tanh(self.att_fc(hr))
            u = u.reshape(h.shape[0], h.shape[1], -1)
            att_applied = F.softmax(torch.matmul(u, self.att_weight), dim=1)
            att_applied = att_applied.squeeze()
        rev_idx = 0
        for rev in revs:
            words = rev["text"].split()
            if len(words) > self.seq_len:
                words = words[:self.seq_len]
            if rev["y"] == 0:
                for i in range(len(words)):
                    word = words[i]
                    if word in nword_num.keys():
                        nword_num[word] += 1
                        nword_weight[word] += att_applied[rev_idx][i].item()
                    else:
                        voca.add(word)
                        nword_num[word] = 1
                        nword_weight[word] = att_applied[rev_idx][i].item()
                        pword_num[word] = 0
                        pword_weight[word] = 0
            else:
                for i in range(len(words)):
                    word = words[i]
                    if word in pword_num.keys():
                        pword_num[word] += 1
                        pword_weight[word] += att_applied[rev_idx][i].item()
                    else:
                        voca.add(word)
                        pword_num[word] = 1
                        pword_weight[word] = att_applied[rev_idx][i].item()
                        nword_num[word] = 0
                        nword_weight[word] = 0
            rev_idx += 1
        for word in voca:
            if nword_num[word] != 0:
                nword_weight[word] /= nword_num[word]
            if pword_num[word] != 0:
                pword_weight[word] /= pword_num[word]
        temp_p = sorted(pword_weight.items(), key=lambda item:item[1], reverse=True)
        temp_n = sorted(nword_weight.items(), key=lambda item:item[1], reverse=True)
        print('POSITIVE:')
        for word, _ in temp_p[0: 50]:
            print('{:15s}    {:.4f}    {:.4f}'.format(word, pword_weight[word], nword_weight[word]))
        print('\n\nNEGATIVE')
        for word, _ in temp_n[0: 50]:
            print('{:15s}    {:.4f}    {:.4f}'.format(word, pword_weight[word], nword_weight[word]))
                        



    def get_idx_from_sent(self, sent, word_idx_map, max_l, k=300):
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


