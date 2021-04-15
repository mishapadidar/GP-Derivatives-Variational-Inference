import math
import numpy as np
import tqdm
import random
import time
from matplotlib import pyplot as plt
from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def count_param_dim(net):
    '''
    count the number of parameters in a net
    '''
    param_total_dim = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            param_total_dim += np.prod(param.data.shape)
    return param_total_dim

def get_loss_dloss(net, loss_fun, data_loader):
    '''
    Given a net, evaluate loss fun and derivatives at current weights
    Output: loss, dloss
    WARNING: If the net uses batchnorm, be careful net.train()
    '''
    net.train() # to activate gradient computations
    net.zero_grad()
    loss_full_data = 0.
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.cuda(), labels.cuda()
        output = net(images)
        loss = loss_fun(output, labels)
        loss_full_data += loss.item()
        loss.backward()
    # collect gradients at each param dimension
    grads = []
    for param in net.parameters():
        grads.append(param.grad.view(-1)/len(data_loader))
    grads = torch.cat(grads)
    loss_full_data /= len(data_loader)
    return loss_full_data, grads

def test_get_loss_dloss(net, loss_fun, data_loader, N=2):
    # select N data, get loss&dloss from differen batch splittings. Should have same loss and dloss
    one_data_train = data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
    one_data_train.data = one_data_train.data[:N]
    one_data_train.targets = one_data_train.targets[:N]
    data_train_loader_1batch = DataLoader(one_data_train, batch_size=N, shuffle=False, num_workers=8)
    loss_full_data_1batch, grads_full_data_1batch = get_loss_dloss(net, loss_fun, data_train_loader_1batch)
    grads_full_data_1batch = np.array(grads_full_data_1batch.cpu())
    
    data_train_loader_Nbatch = DataLoader(one_data_train, batch_size=1, shuffle=False, num_workers=8)
    loss_full_data_Nbatch, grads_full_data_Nbatch = get_loss_dloss(net, loss_fun, data_train_loader_Nbatch)
    grads_full_data_Nbatch = np.array(grads_full_data_Nbatch.cpu())

    loss_single_data_set = []
    grads_single_data_set = []
    for i in range(N):
        one_data_train = data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
        one_data_train.data = one_data_train.data[i:i+1]
        one_data_train.targets = one_data_train.targets[i:i+1]
        data_train_loader = DataLoader(one_data_train, batch_size=1, shuffle=False, num_workers=8)
        loss_single_data, grads_single_data = get_loss_dloss(net, loss_fun, data_train_loader)
        loss_single_data_set.append(loss_single_data)
        grads_single_data_set.append(np.array(grads_single_data.cpu()))

    error_loss_1 = abs(np.mean(loss_single_data_set) - loss_full_data_1batch)
    error_loss_2 = abs(loss_full_data_1batch - loss_full_data_Nbatch)
    error_dloss_1 = abs(np.mean(grads_single_data_set, 0) - grads_full_data_Nbatch).max()
    error_dloss_2 = abs(grads_full_data_Nbatch - grads_full_data_1batch).max()
    print(f"The maximum error in loss is {(max(error_loss_1, error_loss_2)):.4e}.")
    print(f"The maximum error in dloss is {(max(error_dloss_1, error_dloss_2)):.4e}.")

def collect_initial_samples(loss_fun, data_loader, N_sample):
    '''
    Collect data from randomly initialized net: net params, loss and dloss. 
    '''
    loss_set = []
    dloss_set = []
    for i in range(N_sample):
        # randomly generalize a net. Caveat: 
        net = LeNet5().cuda()
        # collect params
        params = {}
        for name, param in net.named_parameters():
            params[name] = param
        # collect loss and dloss
        loss, dloss = get_loss_dloss(net, loss_fun, data_loader)
        loss_set.append(loss)
        dloss_set.append(dloss)
    return params, loss_set, dloss_set

def dict_to_flattened_array(params):
    '''
    convert shape for Bayesian Optimization
    Inputs:
        params: stores param names and param values
    Outputs:
        flattened_array: a flattened array storing param values for the GP model
        info: stores param names and shapes 
    '''
    #TODO
    return flattened_array, info

def flattened_array_to_dict(flattened_array, info):
    '''
    Inverse mapping function of dict_to_flattened_array
    '''
    #TODO
    return params

def train(net, loss_fun, data_loader, epoch):
    '''
    This is a standard training process.
    '''
    #TODO: incorporate our GP model as an optimizer (incorporate new samples and update GP model)
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        output = net(images)
        loss = loss_fun(output, labels)
        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)
        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
        loss.backward()
        optimizer.step()
    return loss_list, batch_list

def test(net, loss_fun, data_loader, n_test):
    '''
    This is a standard testing process. No changes needed.
    '''
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        output = net(images)
        avg_loss += loss_fun(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= n_test
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / n_test))



def main():
    data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
    data_test = MNIST('./data/mnist',
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()]))
    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=False, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    net = LeNet5().cuda()
    loss_fun = nn.CrossEntropyLoss()

    # verify the correctness of sampled data 
    # test_get_loss_dloss(net, loss_fun, data_train_loader, N=40)
    
    # collect N_sample data points
    N_sample = 1000
    params, loss_set, dloss_set = collect_initial_samples(loss_fun, data_train_loader, N_sample=N_sample)
    
    #TODO: store data

if __name__ == '__main__':
    main()