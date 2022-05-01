import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import sys
sys.path.append('../')

from auton_survival import datasets
from auton_survival.preprocessing import Preprocessor


def get_mnist(data_dir='./data/mnist/', batch_size=128):
    train=MNIST(root=data_dir,train=True,download=True)
    test=MNIST(root=data_dir,train=False,download=True)

    X=torch.cat([train.data.float().view(-1,784)/255.,test.data.float().view(-1,784)/255.],0)
    Y=torch.cat([train.targets,test.targets],0)

    dataset=dict()
    dataset['X']=X
    dataset['Y']=Y

    dataloader=DataLoader(TensorDataset(X,Y),batch_size=batch_size,shuffle=True,num_workers=4)

    return dataloader,dataset

def get_support(data_dir="../auton_survival/datasets", batch_size=128):
    outcomes, features = datasets.load_support()
    cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
    num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
	        'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
             'glucose', 'bun', 'urine', 'adlp', 'adls']
    features = Preprocessor().fit_transform(features, cat_feats=cat_feats, num_feats=num_feats)
    x, t, e = features.values, outcomes.time.values, outcomes.event.values

    n = len(x)

    tr_size = int(n*0.70)
    vl_size = int(n*0.10)
    te_size = int(n*0.20)

    x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size+vl_size]

    train_min = x_train.min(axis=0)
    train_max = x_train.max(axis=0)
    val_min = x_val.min(axis=0)
    val_max = x_val.max(axis=0)

    x_train = (x_train-train_min)/(train_max - train_min)
    x_val = (x_val-val_min)/(val_max - val_min)
    # print("x_train", x_train[:10])
    # print("x_train", x_train.min(axis=0), x_train.max(axis=0))
    # convert to float
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    
    return train_loader, None




