import time
import random
from PIL import Image
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import usps

def reform_Ctrain(train_set):
    sel_dat, sel_lab = [], []
    for i in range(len(train_set)):
        if train_set.targets[i] <= 5:
            sel_dat.append(train_set.data[i])
            sel_lab.append(train_set.targets[i])
        elif train_set.targets[i] >= 7:
            sel_dat.append(train_set.data[i])
            sel_lab.append(train_set.targets[i]-1)
        #else:
            #print('6')
    train_set.targets = np.array(sel_lab)
    train_set.data = np.array(sel_dat)
    return train_set

def reform_Strain(train_set):
    sel_dat, sel_lab = [], []
    for i in range(len(train_set)):
        if train_set.labels[i] == 1:
            sel_dat.append(train_set.data[i])
            sel_lab.append(2)
        elif train_set.labels[i] == 2:
            sel_dat.append(train_set.data[i])
            sel_lab.append(1)
        elif train_set.labels[i] >= 8:
            sel_dat.append(train_set.data[i])
            sel_lab.append(train_set.labels[i]-1)
        elif train_set.labels[i] != 7:
            sel_dat.append(train_set.data[i])
            sel_lab.append(train_set.labels[i])
        #else:
            #print(train_set.labels[i])
    train_set.labels = np.array(sel_lab)
    train_set.data = np.array(sel_dat)
    return train_set

def get_testloader(task, batch_size):
    if task == 'm2s':
        dataloader = DataLoader(
            datasets.SVHN('./data/SVHN', split='test', download=True,
                        transform=transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
            batch_size=batch_size, shuffle=False
        )
    elif task == 'm2u':
        dataloader = DataLoader(
            usps.USPS('./data/USPS', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
            batch_size=batch_size, shuffle=False
        )
    elif task == 's2m':
        dataloader = DataLoader(
            datasets.MNIST('./data/',train=False, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32,32)),
                       transforms.Lambda(lambda x: x.convert("RGB")),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                    ])),
            batch_size=batch_size, shuffle=False
        )
    elif task == 's2u':
        dataloader = DataLoader(
            usps.USPS('./data/USPS', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                    ])),
            batch_size=batch_size, shuffle=False
        )
    elif task == 'u2m':
        dataloader = DataLoader(
            dataset = datasets.MNIST('./data/',train=False, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((16,16)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
            batch_size=batch_size, shuffle=False
        )
    elif task == 'u2s':
        dataloader = DataLoader(
            dataset = datasets.SVHN('./data/SVHN', split='test', download=True,
                    transform=transforms.Compose([
                        transforms.Resize((16,16)),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])),
            batch_size=batch_size, shuffle=False
        )
    elif task == 'c2s':
        dataloader = DataLoader(
            dataset = reform_Strain(datasets.STL10('./data/STL', split='test', download=True,
                        transform=transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))),
            batch_size=batch_size, shuffle=False
        )
    elif task == 's2c':
        dataloader = DataLoader(
            dataset = reform_Ctrain(datasets.CIFAR10('./data/CIFAR', train=False, download=True, 
                        transform=transforms.Compose([ 
                            transforms.Resize((96,96)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))),
            batch_size=batch_size, shuffle=False
        )

    return dataloader

def default_loader(path):
        return Image.open(path).convert('L')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,txt, transform=None,target_transform=None, loader=default_loader):
        super(MyDataset,self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader 
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imgs)

def create_target_samples(n=1, task='m2s', clss=10):
    if task == 'm2s':
        dataset = datasets.SVHN('./data/SVHN', split='train', download=True,
                        transform=transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]))
    elif task == 'm2u':
        dataset = usps.USPS('./data/USPS', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize((28,28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    elif task == 's2m':
        dataset = datasets.MNIST('./data/',train=train,download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32,32)),
                       transforms.Lambda(lambda x: x.convert("RGB")),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                   ]))
    elif task == 's2u':
        dataset = usps.USPS('./data/USPS', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                    ]))
    elif task == 'u2m':
        dataset = datasets.MNIST('./data/',train=train,download=True,
                   transform=transforms.Compose([
                       transforms.Resize((16,16)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]))
    elif task == 'u2s':
        dataset = datasets.SVHN('./data/SVHN', split='train', download=True,
                    transform=transforms.Compose([
                        transforms.Resize((16,16)),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))
    elif task == 'c2s':
        dataset = datasets.STL10('./data/STL', split='train', download=True,
                        transform=transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))
        dataset = reform_Strain(dataset)
    elif task == 's2c':
        dataset = datasets.CIFAR10('./data/CIFAR', train=True, download=True, 
                        transform=transforms.Compose([ 
                            transforms.Resize((96,96)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))
        dataset = reform_Ctrain(dataset)
    else:
        print('Warning: Unknown task!')

    X,Y=[],[]
    classes=clss*[n]

    i=0
    #sample_idx = np.random.randint(0,len(dataset),1000)
    while True:
        if len(X)==n*clss:
            break
        x,y=dataset[i]
        #x,y=dataset[sample_idx[i]]
        if classes[y]>0:
            X.append(x)
            Y.append(y)
            classes[y]-=1
        i+=1

    assert (len(X)==n*clss)
    return torch.stack(X,dim=0),torch.from_numpy(np.array(Y))

"""
G1: a pair of pic comes from same domain ,same class
G3: a pair of pic comes from same domain, different classes

G2: a pair of pic comes from different domain,same class
G4: a pair of pic comes from different domain, different classes
"""
def create_groups(X_s,Y_s,X_t,Y_t,seed=1):
    #change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)

    n=X_t.shape[0] #class_num*shot

    #shuffle order
    classes = torch.unique(Y_t)
    classes=classes[torch.randperm(len(classes))]

    class_num=classes.shape[0]
    shot=n//class_num
    
    def s_idxs(c):
        idx=torch.nonzero(Y_s.eq(int(c)))

        return idx[torch.randperm(len(idx))][:shot*2].squeeze()
    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix=torch.stack(source_idxs)
    target_matrix=torch.stack(target_idxs)

    G1, G2, G3, G4 = [], [] , [] , []
    Y1, Y2 , Y3 , Y4 = [], [] ,[] ,[]


    for i in range(class_num):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]],X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]],Y_s[source_matrix[i][j*2+1]]))
            if shot > 1:
                G2.append((X_s[source_matrix[i][j]],X_t[target_matrix[i][j]]))
                Y2.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i][j]]))
            else:
                G2.append((X_s[source_matrix[i][j]],X_t[target_matrix[i]]))
                Y2.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i]]))
            G3.append((X_s[source_matrix[i%class_num][j]],X_s[source_matrix[(i+1)%class_num][j]]))
            Y3.append((Y_s[source_matrix[i % class_num][j]], Y_s[source_matrix[(i + 1) % class_num][j]]))
            if shot > 1:
                G4.append((X_s[source_matrix[i%class_num][j]],X_t[target_matrix[(i+1)%class_num][j]]))
                Y4.append((Y_s[source_matrix[i % class_num][j]], Y_t[target_matrix[(i + 1) % class_num][j]]))
            else:
                G4.append((X_s[source_matrix[i%class_num][j]],X_t[target_matrix[(i+1)%class_num]]))
                Y4.append((Y_s[source_matrix[i % class_num][j]], Y_t[target_matrix[(i + 1) % class_num]]))
    

    groups=[G1,G2,G3,G4]
    groups_y=[Y1,Y2,Y3,Y4]

    #make sure we sampled enough samples
    for g in groups:
        assert(len(g)==n)
    return groups,groups_y

def sample_groups(X_s,Y_s,X_t,Y_t,seed=1):
    #print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,seed=seed)


