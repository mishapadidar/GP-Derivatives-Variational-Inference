from synthetic_functions import *
from rescale import *
import scipy.io
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_synthetic_data(test_fun, n, **kwargs):
    """
    load synthetic data 
    Input: 
        test_fun: a modified Botorch test function
        n: number of datapoints
    Output: 
        x: torch tensor, random data from unit cube
        y: torch tensor, normalized and rescaled labels (w/ or w/o derivatives)
    """
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

    # add scaling factors to info_dict for further accurate plot
    info_dict = {}
    return x_unit, y, info_dict

#use real_helens when calling in exp_script.py
def load_helens(data_src_path, **args):
    """
    load synthetic data 
    Input: 
        data_src_path: path to dataset
        filter_val: float64 in [0, 1]; code will filter out points which possess x-coordinate > filter_val
    Output: 
        train_dataset: torch TensorDataset
        test_dataset: torch TensorDataset
        dim: x-dimension of data
    """
    torch.random.manual_seed(args["seed"])
    n = args["n_train"]
    filter_val = args["filter_val"]
    #n_test = args["n_test"]

    # Apply normalizations to dataset 
    mat = scipy.io.loadmat(data_src_path)
    x = torch.tensor(np.float64(mat['mth_points'])).float()
    SCALE_0_FACTOR = x[:, 0].max()
    SCALE_1_FACTOR = x[:, 1].max()
    x[:, 0] = x[:, 0]/SCALE_0_FACTOR
    x[:, 1] = x[:, 1]/SCALE_1_FACTOR
    y = torch.tensor(np.float64(mat['mth_verts'])).float()
    SCALE_Y_FACTOR = max(y)
    y = y/SCALE_Y_FACTOR
    if args["derivative"]:
        dy = torch.tensor(np.float64(mat['mth_grads'])).float()
        dy = dy / SCALE_Y_FACTOR #modify derivatives due to y-scaling
        dy[:, 0] = dy[:, 0]*SCALE_0_FACTOR #modify derivatives due to x-scaling
        dy[:, 1] = dy[:, 1]*SCALE_1_FACTOR
        data = torch.cat((y, dy), dim = 1).float()
    else:
        data = y
    
    # FILTERING DATA
    # location concatenated with y and dy values, for the sake of filtering
    full_data = torch.cat((x, data), dim=1).float() 
    
    temp_full_data = np.array(full_data)
    def fun(x, val = filter_val):
        if x[0]>val or x[1]>val:       
            return False 
        else:
            return True
    filtered = filter(fun, temp_full_data)
    #for item in filtered:
    #    print(item)
    arr = [item for item in filtered]
    len_arr = len(arr) #number of total points after filtering
    # len_arr = len_arr - len_arr%100
    arr = arr[:len_arr]

    #recover x and data from filtered concatenated values (arr)
    x = torch.tensor([item[0:2] for item in arr])
    data = torch.tensor([item[2:] for item in arr]) 
    data = data.squeeze(-1)   
  
    if torch.cuda.is_available():
        x, data = x.cuda(), data.cuda()
    dataset = TensorDataset(x, data)
    # Train-Test Split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n, len_arr - n])#, generator=torch.Generator().manual_seed(42))
    dim = len(train_dataset[0][0])
    info_dict = {"SCALE_x0_FACTOR": SCALE_0_FACTOR.item(),
                 "SCALE_x1_FACTOR": SCALE_1_FACTOR.item(),
                 "SCALE_Y_FACTOR": SCALE_Y_FACTOR[0].item(),
                 "n_train":n,
                 "n_test": len_arr - n}
    print(info_dict)
    return train_dataset, test_dataset, dim, info_dict


def load_3droad(data_src_path, **args):
    data = torch.Tensor(scipy.io.loadmat(data_src_path)['data'])
    X = data[:, :-2]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]
    y.sub_(y.mean(0)).div_(y.std(0))

    # shuffle the data
    torch.random.manual_seed(args["seed"])
    indices = torch.randperm(X.size(0))[:326155]
    X = X[indices]
    y = y[indices]
    dim = X.shape[-1]

    train_n = args["n_train"]
    # train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()

    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    info_dict = {"n_train":train_n,
                 "n_test": len(X) - train_n}

    return train_dataset, test_dataset, dim, info_dict