import sys
import scipy.io
import torch
import numpy as np 
#sys.path.append("../../directionalvi")
#sys.path.append("../../data")
#sys.path.append("../../directionalvi/utils")

mat = scipy.io.loadmat('MtSH.mat')
#print(mat.keys())
#print(mat['nx'])
#print(mat['ny'])
#print(mat['mth_grads'])
print(mat['mth_points'])
#print(mat['mth_verts'])

x = torch.tensor(np.float64(mat['mth_points']))
y = torch.tensor(np.float64(mat['mth_verts']))
dy = torch.tensor(np.float64(mat['mth_grads']))
data = torch.cat((y, dy), dim = 1)

# data parameters
n   = data.shape[0] 
dim = 2
n_test = n//2

# training params
num_inducing = 30
num_directions = dim
minibatch_size = 200
num_epochs = 1000

# generate training data
train_x = x
train_y = data
train_dataset = TensorDataset(x, data)

# testing data
test_x = torch.rand(n_test,dim)
test_y = f(test_x)
test_dataset = TensorDataset(test_x,test_y)
test_loader = DataLoader(test_dataset, batch_size=n_test, shuffle=False)
