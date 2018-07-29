import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import DAM_model


def train(train_set, dev_set, model, args):
    optimizer = torch.optim.Adam(model.parameters())
    batch_size = args["batch_size"]
    input_num = len(train_set[0])
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
                feature = train_set[0][batch*batch_size:]
                target = train_set[1][batch*batch_size:]
                domain = train_set[2][batch*batch_size:]
            else:
                feature = train_set[0][batch*batch_size:(batch+1)*batch_size]
                target = train_set[1][batch*batch_size:(batch+1)*batch_size]
                domain = train_set[2][batch*batch_size:(batch+1)*batch_size]
            
            feature = torch.tensor(np.array(feature))
            target = torch.tensor(np.array(target))
            domain = torch.tensor(np.array(domain))
            
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
                output, general_output, specific_outputs = model(feature)
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

        ave_acc = ave_acc / len(train_set[0]) * 100
        print('\nEpoch: {} - Train accuracy: {:.4f}%'.format(epoch, ave_acc))
        dev_acc = eval(dev_set, model, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epoch = epoch
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
    feature_all, target_all = data_set[0:2]
    domain_all = []
    if args["data_set"] == "all":
        domain_all = data_set[2]
    feature_all = torch.tensor(np.array(feature_all))
    target_all = torch.tensor(np.array(target_all))
    domain_all = torch.tensor(np.array(domain_all))

    batch_size = args["batch_size"]
    input_num = len(feature_all)
    batch_num = input_num//batch_size
    if input_num % batch_size != 0:
        batch_num = batch_num + 1
    
    if torch.cuda.is_available():
        feature_all = feature_all.cuda(args["GPU"])
        target_all = target_all.cuda(args["GPU"])
        domain_all = domain_all.cuda(args["GPU"])

    ave_loss = 0.0
    total_correct = 0

    for batch in range(0, batch_num):
        if batch == batch_num-1:
            feature = feature_all[batch*batch_size:]
            target = target_all[batch*batch_size:]
            domain = domain_all[batch*batch_size:]
        else:
            feature = feature_all[batch*batch_size:(batch+1)*batch_size]
            target = target_all[batch*batch_size:(batch+1)*batch_size]
            domain = domain_all[batch*batch_size:(batch+1)*batch_size]
        if args["model"] == "dam":
            output, d = model(feature)
            loss = F.cross_entropy(output, target, size_average=False) + args["regular"]*F.cross_entropy(d, domain, size_average=False)
        elif args["model"] == "gs":
            output, general_output, specific_outputs = model(feature)
            loss = model.loss(output, general_output, specific_outputs, target, domain, args["lambda1"], args["lambda2"])*len(feature)
        else:
            output = model(feature)
            loss = F.cross_entropy(output, target, size_average=False)
        correct = (torch.max(output, 1)[1].view(target.size()) == target).sum()
        ave_loss += loss
        total_correct += correct
    size = len(feature_all)
    ave_loss = ave_loss / size
    accuracy = 100.0 * float(total_correct)/size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(ave_loss.item(), 
                                                                       accuracy, 
                                                                       total_correct, 
                                                                       size))
    return accuracy    