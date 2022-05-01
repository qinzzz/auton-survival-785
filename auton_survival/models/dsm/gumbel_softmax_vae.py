# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca), modified by Yongfei Yan
# The code has been modified from pytorch example vae code and inspired by the origianl \
# tensorflow implementation of gumble-softmax by Eric Jang.

import sys
import argparse
import numpy as np
sys.path.append('../')
print(sys.path)

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from auton_survival import datasets

from auton_survival.preprocessing import Preprocessor


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hard', action='store_true', default=False,
                    help='hard Gumbel softmax')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = False

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def load_mnist():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader


def load_support():
    
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

    # x_train = x_train[:, :18]
    # x_val = x_val[:, :18]

    # x_train = (x_train - x_train.mean(axis=0))/x_train.std(axis=0)
    # x_test = (x_test - x_test.mean(axis=0))/x_test.std(axis=0)
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

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=args.batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(x_test, batch_size=args.batch_size, shuffle=True)
    return train_loader, val_loader

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if args.cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    # return F.softmax(y / temperature, dim=-1)
    return y / temperature


def gumbel_softmax(logits, temperature, hard=False, latent_dim=3, categorical_dim=3):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        # return y.view(-1, latent_dim, categorical_dim)
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim, categorical_dim)


class VAE_gumbel(nn.Module):
    def __init__(self, latent_dim=30, categorical_dim=3):
        super(VAE_gumbel, self).__init__()

        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim

        self.fc1 = nn.Linear(38, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 38)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp, hard):
        # q = self.encode(x.view(-1, 784))
        q = self.encode(x)
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        z = gumbel_softmax(q_y, temp, hard, self.latent_dim, self.categorical_dim)
        return self.decode(z), F.softmax(q_y, dim=-1).reshape(*q.size())

    def vae_encode(self, x, temp, hard):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        q = self.relu(self.fc3(h2))

        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        z = gumbel_softmax(q_y, temp, hard, self.latent_dim, self.categorical_dim)
        return z, F.softmax(q_y, dim=-1).reshape(*q.size())

    def vae_decode(self, z):
        z = F.softmax(z, dim=-1).reshape(-1, self.latent_dim * self.categorical_dim)
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy, categorical_dim):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="mean")
    # print("recon_x", recon_x[0])
    # print("x", x[0])
    # print("qy", qy[0])
    # print("BCE loss=", BCE)
    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + 0.01 * KLD


def train(model, epoch, train_loader, optimizer, ANNEAL_RATE, temp_min, categorical_dim):
    model.train()
    train_loss = 0
    bce_loss = 0
    kld_loss = 0
    temp = args.temp
    for batch_idx, data in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()

        recon_batch, qy = model(data, temp, args.hard)
        bce, kld = loss_function(recon_batch, data, qy, categorical_dim)
        print(qy[0])
        loss = bce + kld
        bce_loss += bce.item() * len(data)
        kld_loss += kld.item() * len(data)
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t BCE Loss:{:.6f}\t KLD Loss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item(), bce.item(), kld.item()))
    
    n = data.size(0)
    comparison = torch.cat([data, recon_batch])
    save_image(comparison.data.cpu(),
            '../training/support/train_' + str(epoch) + '.png', nrow=n)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(model, epoch, test_loader, optimizer, ANNEAL_RATE, temp_min, categorical_dim):
    model.eval()
    test_loss = 0
    temp = args.temp
    for i, (data) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        recon_batch, qy = model(data, temp, args.hard)
        bce, kld = loss_function(recon_batch, data, qy, categorical_dim)
        test_loss += (bce + kld).item() * len(data)
        if i % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)
        if i == 0:
            n = data.size(0)
            comparison = torch.cat([data[:n],
                                    recon_batch[:n]])
            save_image(comparison.data.cpu(),
                       '../training/support/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def visualize(model, test_loader):
    model.train()
    train_loss = 0
    temp = args.temp
    for batch_idx, data in enumerate(test_loader):
        # print("batch idx", batch_idx)
        if args.cuda:
            data = data.cuda()
        recon_batch, qy = model(data, temp, args.hard)
        print("original data", data[:2])
        print("recon batch", recon_batch[:2])
        input()

def run():
    latent_dim = 10
    categorical_dim = 6

    temp_min = 0.5
    ANNEAL_RATE = 0.00003

    model = VAE_gumbel(latent_dim, categorical_dim)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # train_loader, test_loader = load_mnist()
    train_loader, test_loader = load_support()

    for epoch in range(1, args.epochs + 1):
        train(model, epoch, train_loader, optimizer, ANNEAL_RATE, temp_min, categorical_dim)
        test(model, epoch, test_loader, optimizer, ANNEAL_RATE, temp_min, categorical_dim)

    M = 64 * latent_dim
    np_y = np.zeros((M, categorical_dim), dtype=np.float32)
    np_y[range(M), np.random.choice(categorical_dim, M)] = 1
    np_y = np.reshape(np_y, [M // latent_dim, latent_dim, categorical_dim])
    sample = torch.from_numpy(np_y).view(M // latent_dim, latent_dim * categorical_dim)
    # print("random", sample[:10])
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(M // latent_dim, -1),
                '../training/support/sample_' + str(epoch) + '.png')
    
    # generate for each class
    # for i in range(categorical_dim):
    #     M = 64 * latent_dim
    #     np_y = np.zeros((M, categorical_dim), dtype=np.float32)
    #     np_y[range(M), i] = 1
    #     np_y = np.reshape(np_y, [M // latent_dim, latent_dim, categorical_dim])
    #     sample = torch.from_numpy(np_y).view(M // latent_dim, latent_dim * categorical_dim)
    #     # print(i, sample[:10])
    #     if args.cuda:
    #         sample = sample.cuda()
    #     sample = model.decode(sample).cpu()
    #     save_image(sample.data.view(M // latent_dim, -1),
    #                 f'../training/support/sample_class_{i+1}.png')

if __name__ == '__main__':
    run()
