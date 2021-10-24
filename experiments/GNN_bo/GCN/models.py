import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def update_weights(self,weights):
        """
        method to update the weights of the nn
        """
        # dont track update in grad
        self.eval()

        # ordered keys of params
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        keys.sort() # ensure we have the same order each time

        used_params = 0
        #for key in keys:
        for param in self.parameters():
            # get the size and shape of the parameter
            #param_size = state_dict[key].numel()
            #param_shape = state_dict[key].shape
            param_size = param.numel()
            param_shape = param.shape
            new_params = weights[used_params:used_params+param_size].reshape(param_shape)
            # Update the parameter.
            #state_dict[key].copy_(new_params)
            param.data = new_params
            # counter
            used_params +=param_size

    def get_grad(self):
        grads = []
        for param in self.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)

        return grads

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
