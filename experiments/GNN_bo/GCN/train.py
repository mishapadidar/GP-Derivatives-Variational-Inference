from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import *
from models import GCN
try: # import wandb if watch model on weights&biases
  import wandb
except:
  pass

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--n_train', type=int, default=10000,
                    help='number of training data')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='batch size')
parser.add_argument('--train_percent', type=float, default=0.1,
                    help='training label percentage')
parser.add_argument('--watch_model', type=bool, default=False,
                    help='watch model from wandb')
parser.add_argument('--expid', type=int, default="-",
                    help='experiment id')
parser.add_argument('--lr_sched', type=str, default="step_lr",
                    help='type of learning rate scheduler')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.watch_model: # watch model on weights&biases
        wandb.init(project='L2C', entity='xinranzhu',
                name=f"{args.dataset}_exp{args.expid}")
        print("Experiment settings:")
        print(args)
        wandb.config.seed = args.seed
        wandb.config.dropout = args.dropout
        wandb.config.epochs = args.epochs
        wandb.config.lr = args.lr
        wandb.config.weight_decay = args.weight_decay
        wandb.config.hidden = args.hidden
        wandb.config.train_percent = args.train_percent
        wandb.config.expid = args.expid


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.dataset == "cora":
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset, 
                                                                    train_percent=args.train_percent)
if args.dataset == "reddit":
    adj, features, labels, idx_train, idx_val, idx_test = load_reddit(args.n_train)

if args.dataset == "citeseer":
    adj, features, labels, idx_train, idx_val, idx_test = load_citeseer(train_percent=args.train_percent)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

if args.lr_sched == "step_lr":
    milestones = [int(len(idx_train)/3), int(2*len(idx_train)/3)]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
elif args.lr_sched == "lambda_lr":
    lr_sched_fun = lambda epoch: 1.0/(epoch+1)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched_fun)
else:
    lr_scheduler = None

def train(epoch, print_loss=True, lr_scheduler=None):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    wandb.log({'loss': loss_train})
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if lr_scheduler != None:
        variational_scheduler.step()
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if print_loss:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    print_loss = True if epoch % 50 == 0 else False
    train(epoch, print_loss=print_loss)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
wandb.save(f"a.out_{args.dataset}exp{args.expid}")

