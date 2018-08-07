import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import DAM_model


def train(train_set, dev_set, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    batch_size = args["batch_size"]
    input_num = train_set.shape[0]
    batch_num = input_num//batch_size
    if input_num % batch_size != 0:
        batch_num = batch_num + 1
    best_epoch = 0
    best_acc = 0
    dec_count = 0
    last_acc = 0
    for epoch in range(0, args["epoch_num"]):
        model.train()
        ave_acc = 0.0
        for batch in range(0, batch_num):

            if batch == batch_num-1:
                feature = train_set[batch*batch_size:, :-2]
                target = train_set[batch*batch_size:, -2]
                domain = train_set[batch*batch_size:, -1]
            else:
                feature = train_set[batch*batch_size:(batch+1)*batch_size, :-2]
                target = train_set[batch*batch_size:(batch+1)*batch_size, -2]
                domain = train_set[batch*batch_size:(batch+1)*batch_size, -1]
            
            feature = torch.tensor(feature)
            target = torch.tensor(target)
            domain = torch.tensor(domain)
            
            if torch.cuda.is_available():
                feature = feature.cuda(args["GPU"])
                target = target.cuda(args["GPU"])
                domain = domain.cuda(args["GPU"])

            if args["model"] == "dam":
                optimizer.zero_grad()
                output, d = model(feature)
                loss = F.cross_entropy(output, target) + args["regular"]*F.cross_entropy(d, domain)
                loss.backward()
                optimizer.step()
            elif args["model"] == "gs":
                optimizer.zero_grad()
                output, general_output, specific_outputs = model(feature, domain)
                loss = model.loss(output, general_output, specific_outputs, target, domain, args["lambda1"], args["lambda2"])
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                output = model(feature)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            corrects = (torch.max(output, 1)[1].view(target.size()) == target).sum()
            accuracy = 100.0 * float(corrects)/len(feature)
            ave_acc = ave_acc + float(corrects)
            sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(batch+epoch*batch_num, loss.item(), accuracy, corrects, len(feature)))


        ave_acc = ave_acc / train_set.shape[0] * 100
        print('\nEpoch: {} - Train accuracy: {:.4f}%'.format(epoch, ave_acc))
        dev_acc = eval(dev_set, model, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epoch = epoch
            if args["model"] == "gs":
                torch.save(model, "gs"+"_hd_"+str(args["attention_dim"])+"_l1_"+str(args["lambda1"])+"_l2_"+str(args["lambda2"])+".model")
            else:
                torch.save(model, args["data_set"] + '_' + args["model"] + '.model')

        if dev_acc < last_acc:
            dec_count = dec_count + 1
        else:
            dec_count = 0
        last_acc = dev_acc
        if dec_count >= args["early_stop"]:
            break
        # elif ave_acc > 95:
        #     break
        # elif dev_acc < best_acc - 5:
        #     break
    print('best epoch:{}    best accuracy:{:.4f}%'.format(best_epoch, best_acc))
    return best_acc



def eval(data_set, model, args):
    model.eval()
    with torch.no_grad():
        feature = torch.tensor(data_set[:, :-2])
        target = torch.tensor(data_set[:, -2])
        domain = torch.tensor(data_set[:, -1])
        
        if torch.cuda.is_available():
            feature = feature.cuda(args["GPU"])
            target = target.cuda(args["GPU"])
            domain = domain.cuda(args["GPU"])

        if args["model"] == "dam":
            output, d = model(feature)
            loss = F.cross_entropy(output, target) + args["regular"]*F.cross_entropy(d, domain)
        elif args["model"] == "gs":
            output, general_output, specific_outputs = model(feature, domain)
            loss = model.loss(output, general_output, specific_outputs, target, domain, args["lambda1"], args["lambda2"])
        else:
            output = model(feature)
            loss = F.cross_entropy(output, target)
        correct = (torch.max(output, 1)[1].view(target.size()) == target).sum()
        size = len(feature)
        accuracy = 100.0 * float(correct)/size
        print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(loss.item(), 
                                                                        accuracy, 
                                                                        correct, 
                                                                        size))
    return accuracy    