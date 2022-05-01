import argparse
from dataloader import *
from model import VaDE, cluster_acc
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import torch.nn as nn


if __name__ == '__main__':

    parse=argparse.ArgumentParser(description='VaDE')
    parse.add_argument('--lr',type=float,default=1e-4)
    parse.add_argument('--batch_size',type=int,default=128)
    parse.add_argument('--datadir',type=str,default='./data/mnist')
    parse.add_argument('--nClusters',type=int,default=3)
    parse.add_argument('--epochs',type=int,default=100)
    parse.add_argument('--hid_dim',type=int,default=10)
    parse.add_argument('--cuda',type=bool,default=False)

    args=parse.parse_args()

    # DL, _ = get_mnist(args.datadir,args.batch_size)
    DL, _ = get_support(args.datadir,args.batch_size)

    vade=VaDE(args)
    if args.cuda:
        vade=vade.cuda()
        vade=nn.DataParallel(vade,device_ids=range(4))

    vade.pre_train(DL,pre_epoch=50)

    opti=Adam(vade.parameters(),lr=args.lr)
    lr_s=StepLR(opti,step_size=10,gamma=0.95)

    writer=SummaryWriter('./logs')
    epoch_bar=tqdm(range(args.epochs))
    tsne=TSNE()

    for epoch in epoch_bar:
        lr_s.step()
        L=0
        for x in DL:
            if args.cuda:
                x=x.cuda()
            loss=vade.ELBO_Loss(x)
            
            opti.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vade.parameters(), 1.0)
            opti.step()
            L+=loss.detach().cpu().numpy()

        input_data = []
        pred=[]
        # tru=[]
        with torch.no_grad():
            for x in DL:
                if args.cuda:
                    x = x.cuda()

                # tru.append(y.numpy())
                input_data.append(x)
                pred.append(vade.predict(x))

        writer.add_scalar('loss',L/len(DL),epoch)
        # writer.add_scalar('acc',cluster_acc(pre,tru)[0]*100,epoch)
        writer.add_scalar('lr',lr_s.get_lr()[0],epoch)
        print("predicted:", pred)
        epoch_bar.write('Loss={:.4f},LR={:.4f}'.format(L/len(DL), lr_s.get_lr()[0]))
    
    xs = []
    preds = []
    for x in DL:
        pred = vade.predict(x)
       
        preds.append(pred)
        xs.append(x)

    xs = np.concatenate(xs, 0)
    preds = np.concatenate(preds, 0)
    print("xs", xs.shape)
    print("preds", preds.shape)

    with open(f"pred_clusters_{args.lr}_{args.batch_size}_{args.nClusters}.npy", "wb") as f:
        np.save(f, xs)
        np.save(f, preds)







