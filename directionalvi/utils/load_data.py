from synthetic_functions import *
from rescale import *
import scipy.io
from torch.utils.data import TensorDataset, DataLoader

def load_synthetic_data(test_fun, n, **kwargs):
    '''
    load synthetic data 
    Input: 
        test_fun: a modified Botorch test function
        n: number of datapoints
    Output: 
        x: torch tensor, random data from unit cube
        y: torch tensor, normalized and rescaled labels (w/ or w/o derivatives)
    '''
    torch.random.manual_seed(kwargs["seed"])
    dim = test_fun.dim
    x_unit = torch.rand(n,dim)
    # evaluate in the true range
    lb, ub = test_fun.get_bounds()
    x = from_unit_cube(x_unit, lb, ub)
    if kwargs["derivative"]:
        y = test_fun.evaluate_true_with_deriv(x)
    else:
        y = test_fun.evaluate_true(x)
    # normalize y values (with or without derivatives)
    normalize(y, **kwargs)
    if kwargs["derivative"]:
        # mapping derivative values to unit cube
        f = y[..., 0].reshape(len(y),1)
        g = y[..., 1:].reshape(len(y),-1)
        g *= (ub - lb)
        y = torch.cat([f, g], 1)
    return x_unit, y

#use real_helens when calling in exp_script.py
def load_helens(data_src_path, **args):
    '''
    load synthetic data 
    Input: 
        data_src_path: path to dataset
    Output: 
        train_dataset: torch TensorDataset
        test_dataset: torch TensorDataset
        dim: x-dimension of data
    '''
    torch.random.manual_seed(args["seed"])
    n = args["n_train"]
    n_test = args["n_test"]

    # Apply normalizations to dataset 
    mat = scipy.io.loadmat(data_src_path)
    x = torch.tensor(np.float64(mat['mth_points'])).float()
    SCALE_0_FACTOR = max(x[:, 0])
    SCALE_1_FACTOR = max(x[:, 1])
    x[:, 0] = x[:, 0]/SCALE_0_FACTOR
    x[:, 1] = x[:, 1]/SCALE_1_FACTOR
    y = torch.tensor(np.float64(mat['mth_verts'])).float()
    SCALE_Y_FACTOR = max(y)
    y = y/SCALE_Y_FACTOR
    dy = torch.tensor(np.float64(mat['mth_grads'])).float()
    dy = dy / SCALE_Y_FACTOR #modify derivatives due to y-scaling
    dy[:, 0] = dy[:, 0]*SCALE_0_FACTOR #modify derivatives due to x-scaling
    dy[:, 1] = dy[:, 1]*SCALE_1_FACTOR
    data = torch.cat((y, dy), dim = 1).float()
    #full_data = torch.cat((x, data), dim=1).float() #location concatenated with y and dy values
    if torch.cuda.is_available():
        x, data = x.cuda(), data.cuda()
    dataset = TensorDataset(x, data)

    # Train-Test Split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n, n_test])#, generator=torch.Generator().manual_seed(42))
    dim = len(train_dataset[0][0])

    return train_dataset, test_dataset, dim