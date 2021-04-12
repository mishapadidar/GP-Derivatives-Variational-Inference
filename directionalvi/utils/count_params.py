import numpy as np

def count_params(model, likelihood):
    # count number of parameters to learn
    param_total_dim = 0
    print("All parameters to learn:")
    for name, param in model.named_parameters():
        print("     ", name)
        print("     ", param.data.shape)
        if param.requires_grad:
            param_total_dim += np.prod(param.data.shape)
    for name, param in likelihood.named_parameters():
        print("     ", name)
        print("     ", param.data.shape)
        if param.requires_grad:
            param_total_dim += np.prod(param.data.shape)

    print("Total number of parameters: ", param_total_dim)
    return param_total_dim