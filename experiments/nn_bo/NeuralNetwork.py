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
            #seq.append(nn.ReLU())
            seq.append(nn.Sigmoid())
        seq.append(nn.Linear(layer_height,1,bias=False))
        # make a stack
        self.lin_relu_stack = nn.Sequential(*seq)
        # count num params.... Warning: it may count non-trainable params
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.lin_relu_stack(x)


    def update_weights(self,weights):
        """
        method to update the weights of the nn
        """
        assert len(weights) == self.n_params
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

    def zero_grad(self):
        """zero out the grads. 
           Call this before next prediction,loss, grad computation"""
        for param in self.parameters():
            param.grad.data.zero_()

    def get_grad(self):
        grads = []
        for param in self.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)

        return grads




if __name__ == "__main__":
  dim = 18
  n_hidden_layers = 3
  n_layers = n_hidden_layers+1
  random_data = torch.rand(dim)
  my_nn = NeuralNetwork(n_hidden_layers,dim)
  print(my_nn.n_params)
  my_nn.train()
  result = my_nn(random_data)
  print(result)
  # compute the loss
  mse_loss = nn.MSELoss()
  y = torch.randn(1)
  output = mse_loss(result,y)
  print('loss: ',output)
  output.backward()
  grad = my_nn.get_grad() 
  print('grad: ',grad)
  weights = torch.randn(my_nn.n_params)
  my_nn.update_weights(weights)
  result = my_nn(random_data)
  print(result)
