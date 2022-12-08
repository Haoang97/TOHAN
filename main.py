import os, time
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from main_models import DCD, Encoder_1c, Encoder_3c, Classifier
import argparse
import dataloader
from generotor import generator_1c, generator_3c
from torch.autograd import Variable

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='m2s')
parser.add_argument('--gd_epoch',type=int,default=100)
parser.add_argument('--model_epoch',type=int,default=50)
parser.add_argument('--generate_epoch',type=int,default=500)
parser.add_argument('--n_target_samples',type=int,default=7)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--generate_batch',type=int,default=32)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--generator_dim',type=int,default=64)
parser.add_argument('--gd_dim',type=int,default=128)
parser.add_argument('--data_size',type=int,default=28)
parser.add_argument('--data_channel',type=int,default=1)
parser.add_argument('--lambda',type=float,default=0.2)

opt=vars(parser.parse_args())

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')

torch.manual_seed(opt['seed'])
if use_cuda:
    torch.cuda.manual_seed(opt['seed'])

l1loss = nn.L1Loss().to(device)
MSELoss = nn.MSELoss().to(device)

def metric(G_result, X, n, batch_size):
    all_sum =0
    for i in range(batch_size):
        sum_abs_diff = 0
        for j in range(n):
            abs_diff = torch.abs(G_result[i].view(-1) - X[j].view(-1))
            w = abs_diff / torch.norm(abs_diff, p=2, keepdim=False)
            sum_abs_diff += torch.dot(w, abs_diff)
        sum_abs_diff = sum_abs_diff/n
        all_sum += sum_abs_diff
    all_sum = all_sum/batch_size
    return all_sum

def get_single_target(X_t, Y_t, target):
    X = []
    for i in range(len(X_t)):
        if Y_t[i] == target:
            X.append(X_t[i])
    X = torch.tensor([item.cpu().detach().numpy() for item in X]).to(device)
    return X

def generator_loss(G_result, X_t, Y_t, num, target, const, batch_size):
    X = get_single_target(X_t, Y_t, target)
    all_sum = metric(G_result, X, num, batch_size)
    logits = classifier(encoder(G_result))
    logits = logits.to(device)
    ones = torch.ones(batch_size)
    ones = ones.to(device)
    G_train_loss = MSELoss(logits[:,target], ones) + const * all_sum
    return G_train_loss

def creat_optimizer(target):
    return optim.Adam(G[str(target)].parameters(), lr=opt['lr']*0.1, betas=(0.5, 0.999))

def beta(epoch):
    return 2/(1+np.exp(-10*epoch))-1

if opt['task'] in ['c2s','s2c']:
    classes = 9
else:
    classes = 10
#-------------------model & dir---------------------------------
if opt['task'] in ['m2s', 'm2u']:
    encoder = Encoder_1c()
    classifier = Classifier(input_features=64,outdim=10)
    encoder_dir = './model/mnist_Encoder.pt'
    classifier_dir = './model/mnist_Classifier.pt'
elif opt['task'] in ['s2m','s2u']:
    encoder = Encoder_3c()
    classifier = Classifier(input_features=64,outdim=10)
    encoder_dir = './model/svhn_Encoder.pt'
    classifier_dir = './model/svhn_Classifier.pt'
elif opt['task'] in ['u2m','u2s']:
    encoder = Encoder_1c()
    classifier = Classifier(input_features=64,outdim=10)
    encoder_dir = './model/usps_Encoder.pt'
    classifier_dir = './model/usps_Classifier.pt'
elif opt['task'] == 'c2s':
    encoder = Encoder_3c()
    classifier = Classifier(input_features=64,outdim=9)
    encoder_dir = './model/cifar_Encoder.pt'
    classifier_dir = './model/cifar_Classifier.pt'
elif opt['task'] == 's2c':
    encoder = Encoder_3c()
    classifier = Classifier(input_features=64,outdim=9)
    encoder_dir = './model/stl_Encoder.pt'
    classifier_dir = './model/stl_Classifier.pt'
else:
    print('Warning: Unknown task!')
#-------------------load source model--------------------------
encoder.load_state_dict(torch.load(encoder_dir))
encode = encoder.to(device)

classifier.load_state_dict(torch.load(classifier_dir))
classifier = classifier.to(device)

#------------------Sample target data--------------------------
X_t,Y_t = dataloader.create_target_samples(opt['n_target_samples'], opt['task'], classes)

#------------------generate fake data & FSDA--------------------------
if opt['task'] in ['m2s','m2u','u2m','u2m']:
    G_0 = generator_1c(opt['generator_dim']);G_1 = generator_1c(opt['generator_dim']);G_2 = generator_1c(opt['generator_dim']);G_3 = generator_1c(opt['generator_dim']);G_4 = generator_1c(opt['generator_dim'])
    G_5 = generator_1c(opt['generator_dim']);G_6 = generator_1c(opt['generator_dim']);G_7 = generator_1c(opt['generator_dim']);G_8 = generator_1c(opt['generator_dim']);G_9 = generator_1c(opt['generator_dim'])
else:
    G_0 = generator_3c(opt['generator_dim']);G_1 = generator_3c(opt['generator_dim']);G_2 = generator_3c(opt['generator_dim']);G_3 = generator_3c(opt['generator_dim']);G_4 = generator_3c(opt['generator_dim'])
    G_5 = generator_3c(opt['generator_dim']);G_6 = generator_3c(opt['generator_dim']);G_7 = generator_3c(opt['generator_dim']);G_8 = generator_3c(opt['generator_dim']);G_9 = generator_3c(opt['generator_dim'])
    
G_0.weight_init(mean=0.0, std=0.02);G_1.weight_init(mean=0.0, std=0.02);G_2.weight_init(mean=0.0, std=0.02);G_3.weight_init(mean=0.0, std=0.02)
G_4.weight_init(mean=0.0, std=0.02);G_5.weight_init(mean=0.0, std=0.02);G_6.weight_init(mean=0.0, std=0.02);G_7.weight_init(mean=0.0, std=0.02)
G_8.weight_init(mean=0.0, std=0.02);G_9.weight_init(mean=0.0, std=0.02)

G_0 = G_0.to(device);G_1 = G_1.to(device);G_2 = G_2.to(device);G_3 = G_3.to(device);G_4 = G_4.to(device)
G_5 = G_5.to(device);G_6 = G_6.to(device);G_7 = G_7.to(device);G_8 = G_8.to(device);G_9 = G_9.to(device)

G = {'0':G_0,'1':G_1,'2':G_2,'3':G_3,'4':G_4,'5':G_5,'6':G_6,'7':G_7,'8':G_8,'9':G_9}

G_optimizer = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
for target in range(classes):
    G_optimizer[str(target)] = creat_optimizer(target)

G_losses = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}

save_acc =[]

test_dataloader = dataloader.get_testloader(opt['task'],batch_size=opt['batch_size'])

for epoch in range(1, opt['generate_epoch']+1):
    encoder = encoder.eval()
    classifier = classifier.eval()
    X_s = torch.Tensor(classes*opt['generate_batch'],opt['data_channel'],opt['data_size'],opt['data_size'])
    Y_s = torch.LongTensor(classes*opt['generate_batch'])

    for target in range(classes):
        z_ = torch.randn(opt['generate_batch'], 100).view(-1, 100, 1, 1)
        z_ = Variable(z_.to(device))
        G_optimizer[str(target)].zero_grad()

        G_result = G[str(target)](z_)
        G_result = G_result.to(device)
        logits = classifier(encoder(G_result))

        G_losses[str(target)] = generator_loss(G_result=G_result, X_t=X_t, Y_t=Y_t, num=opt['n_target_samples'], target=target, const= opt['lambda'], batch_size=opt['generate_batch'])
        G_losses[str(target)].backward()
        G_optimizer[str(target)].step()

        X = G_result
        Y = target * torch.ones(opt['generate_batch'], dtype = torch.uint8)
        X_s[target*opt['generate_batch']:(target+1)*opt['generate_batch']] = X
        Y_s[target*opt['generate_batch']:(target+1)*opt['generate_batch']] = Y
        
        random_num = np.random.randint(0,opt['generate_batch'],1)
        if epoch%100 == 0:
            print('[%d/%d]    target: %d    logits: %.5f    loss_g: %.3f' % (epoch, opt['generate_epoch'], target,
             logits[random_num,target].item(), G_losses[str(target)]))

    if epoch == opt['generate_epoch']-opt['model_epoch']:
        index = torch.randperm(classes*opt['generate_batch'])
        X_s = X_s[index]
        Y_s = Y_s[index]

        discriminator = DCD(input_features=opt['gd_dim'])
        discriminator = discriminator.to(device)
        discriminator.train()

        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=opt['lr']*0.1)

        for epoch_2 in range(opt['gd_epoch']):
            # data
            groups,aa = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=epoch_2)

            n_iters = 4 * len(groups[1])
            index_list = torch.randperm(n_iters)
            mini_batch_size=40 #use mini_batch train can be more stable

            loss_mean=[]

            X1=[];X2=[];ground_truths=[]

            for index in range(n_iters):
                ground_truth=index_list[index]//len(groups[1])

                x1,x2=groups[ground_truth][index_list[index]-len(groups[1])*ground_truth]
                X1.append(x1)
                X2.append(x2)
                ground_truths.append(ground_truth)

                #select data for a mini-batch to train
                if (index+1)%mini_batch_size==0:
                    X1=torch.stack(X1)
                    X2=torch.stack(X2)
                    ground_truths=torch.LongTensor(ground_truths)
                    X1=X1.to(device)
                    X2=X2.to(device)
                    ground_truths=ground_truths.to(device)

                    optimizer_D.zero_grad()
                    X_cat=torch.cat([encoder(X1),encoder(X2)],1)
                    y_pred=discriminator(X_cat.detach())
                    loss=loss_fn(y_pred,ground_truths)
                    loss.backward()
                    optimizer_D.step()
                    loss_mean.append(loss.item())
                    X1 = []
                    X2 = []
                    ground_truths = []
            print("pretrain group discriminator----Epoch %d/%d loss:%.3f"%(epoch_2+1,opt['gd_epoch'],np.mean(loss_mean)))

    if epoch > opt['generate_epoch']-opt['model_epoch']:
        encoder.train()
        classifier.train()
        discriminator.eval()
        index = torch.randperm(classes*opt['generate_batch'])
        X_s = X_s[index]
        Y_s = Y_s[index]
        
        optimizer_g_h=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=opt['lr'])
        optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=opt['lr']*0.1)

        scheduler_g_h=torch.optim.lr_scheduler.StepLR(optimizer_g_h,step_size=20,gamma=0.1)
        scheduler_d=torch.optim.lr_scheduler.StepLR(optimizer_d,step_size=20,gamma=0.1)

        #---training g and h , group discriminator is frozen

        groups, groups_y = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=opt['gd_epoch']+epoch)
        G1, G2, G3, G4 = groups
        Y1, Y2, Y3, Y4 = groups_y

        groups_2 = [G2, G4]
        groups_y_2 = [Y2, Y4]

        n_iters = 2 * len(G2)
        index_list = torch.randperm(n_iters)

        n_iters_dcd = 4 * len(G2)
        index_list_dcd = torch.randperm(n_iters_dcd)

        mini_batch_size_g_h = 20 #data only contains G2 and G4 ,so decrease mini_batch
        mini_batch_size_dcd= 40 #data contains G1,G2,G3,G4 so use 40 as mini_batch
        X1 = []
        X2 = []
        ground_truths_y1 = []
        ground_truths_y2 = []
        dcd_labels=[]
        g_h_loss_mean=[]
        d_loss_mean=[]
        for index in range(n_iters):
            ground_truth=index_list[index]//len(G2)
            x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]

            dcd_label=0 if ground_truth==0 else 2
            X1.append(x1.detach())
            X2.append(x2.detach())
            ground_truths_y1.append(y1.detach())
            ground_truths_y2.append(y2.detach())
            dcd_labels.append(dcd_label)

            if (index+1)%mini_batch_size_g_h==0:

                X1=torch.stack(X1)
                X2=torch.stack(X2)
                ground_truths_y1=torch.LongTensor(ground_truths_y1)
                ground_truths_y2 = torch.LongTensor(ground_truths_y2)
                dcd_labels=torch.LongTensor(dcd_labels)
                X1=X1.to(device)
                X2=X2.to(device)
                ground_truths_y1=ground_truths_y1.to(device)
                ground_truths_y2 = ground_truths_y2.to(device)
                dcd_labels=dcd_labels.to(device)

                optimizer_g_h.zero_grad()

                encoder_X1=encoder(X1)
                encoder_X2=encoder(X2)

                X_cat=torch.cat([encoder_X1,encoder_X2],1)
                y_pred_X1=classifier(encoder_X1)
                y_pred_X2=classifier(encoder_X2)
                y_pred_dcd=discriminator(X_cat)

                #loss_X1=loss_fn(y_pred_X1,ground_truths_y1)
                loss_X2=loss_fn(y_pred_X2,ground_truths_y2)
                loss_dcd=loss_fn(y_pred_dcd,dcd_labels)

                loss_sum =  loss_X2 + beta(epoch-opt['generate_epoch']+opt['gd_epoch']) * loss_dcd #

                loss_sum.backward()
                g_h_loss_mean.append(loss_sum.item())
                optimizer_g_h.step()
                scheduler_g_h.step()

                X1 = []
                X2 = []
                ground_truths_y1 = []
                ground_truths_y2 = []
                dcd_labels = []
                
        #----training group discriminator ,g and h frozen
        encoder.eval()
        classifier.eval()
        discriminator.train()
        X1 = []
        X2 = []
        ground_truths = []
        for index in range(n_iters_dcd):

            ground_truth=index_list_dcd[index]//len(groups[1])

            x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            if (index + 1) % mini_batch_size_dcd == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_d.zero_grad()
                X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
                y_pred = discriminator(X_cat.detach())
                loss_d = loss_fn(y_pred, ground_truths)
                loss_d.backward()
                optimizer_d.step()
                scheduler_d.step()
                d_loss_mean.append(loss_d.item())
                X1 = []
                X2 = []
                ground_truths = []
        #testing
        acc = 0
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            for data, labels in test_dataloader:
                data = data.to(device)
                labels = labels.to(device)
                y_test_pred = classifier(encoder(data))
                acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

        accuracy = round(acc / float(len(test_dataloader)), 3)
        save_acc.append(accuracy)
        print("step3----Epoch %d/%d    g_h_loss: %.3f    d_loss: %.3f    accuracy: %.3f " % (epoch, opt['generate_epoch'],np.mean(g_h_loss_mean),np.mean(d_loss_mean), 100*accuracy))