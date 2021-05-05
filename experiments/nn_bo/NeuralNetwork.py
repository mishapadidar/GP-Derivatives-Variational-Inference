import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self,n_hidden_layers,layer_height):
        super(NeuralNetwork, self).__init__()
        # numer of nodes in a layer
        self.layer_height = layer_height 
        # number of hidden layers
        self.n_hidden_layers = n_hidden_layers 
        # sequence of nn operations
        seq =[]
        for ii in range(n_hidden_layers):
            seq.append(nn.Linear(layer_height,layer_height,bias=False))
            seq.append(nn.ReLU())
        seq.append(nn.Linear(layer_height,1,bias=False))
        # make a stack
        self.lin_relu_stack = nn.Sequential(*seq)
        # count num params.... Warning: it may count non-trainable params
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        self.eval()
        return self.lin_relu_stack(x)


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
        for key in keys:
            # Don't update if this is not a weight.
            if not "weight" in key:
                continue
            # get the size and shape of the parameter
            param_size = state_dict[key].numel()
            param_shape = state_dict[key].shape
            new_params = weights[used_params:used_params+param_size].reshape(param_shape)
            # Update the parameter.
            state_dict[key].copy_(new_params)
            # counter
            used_params +=param_size


if __name__ == "__main__":
  dim = 4
  n_hidden_layers = 2
  n_layers = n_hidden_layers+1
  random_data = torch.rand(dim)
  my_nn = NeuralNetwork(n_hidden_layers,dim)
  result = my_nn(random_data)
  print(result)
  weights = torch.randn(my_nn.n_params)
  my_nn.update_weights(weights)
  result = my_nn(random_data)
  print(result)
