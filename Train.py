import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


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
            else:
                feature = train_set[0][batch*batch_size:(batch+1)*batch_size]
                target = train_set[1][batch*batch_size:(batch+1)*batch_size]
            
            feature = np.array(feature)
            target = np.array(target)
            feature = torch.tensor(feature)
            target = torch.tensor(target)

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
        if dev_acc >= best_acc:
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



def eval(data_set, model, args):
    model.eval()
    feature, target = data_set
    feature = np.array(feature)
    target = np.array(target)
    feature = torch.tensor(feature)
    target = torch.tensor(target)
    
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