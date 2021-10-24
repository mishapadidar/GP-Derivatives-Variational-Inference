import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx


# https://github.com/praxidike97/GraphNeuralNet/blob/master/main.py

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 8)
        self.conv2 = GCNConv(8, dataset.num_classes)
        self.n_params = sum(p.numel() for p in self.parameters())
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

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
