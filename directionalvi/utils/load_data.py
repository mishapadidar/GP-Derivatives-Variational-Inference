from synthetic_functions import *
from rescale import *
import scipy.io
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
    return x_unit, y

#use real_helens when calling in exp_script.py
def load_helens(data_src_path, filter_val, **args):
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
        data = y.squeeze(1)
    
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
    len_arr = len_arr - len_arr%100
    arr = arr[:len_arr]

    #recover x and data from filtered concatenated values (arr)
    x = torch.tensor([item[0:2] for item in arr])
    data = torch.tensor([item[2:] for item in arr])    
  
    if torch.cuda.is_available():
        x, data = x.cuda(), data.cuda()
    dataset = TensorDataset(x, data)

    # Train-Test Split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n, len_arr - n])#, generator=torch.Generator().manual_seed(42))
    dim = len(train_dataset[0][0])
    info_dict = {"SCALE_x0_FACTOR": SCALE_0_FACTOR.item(),
                 "SCALE_x1_FACTOR": SCALE_1_FACTOR.item(),
                 "SCALE_Y_FACTOR": SCALE_Y_FACTOR[0].item()}

    return train_dataset, test_dataset, dim, info_dict