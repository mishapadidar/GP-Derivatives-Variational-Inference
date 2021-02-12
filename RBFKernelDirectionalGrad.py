import torch

from gpytorch.lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from gpytorch.kernels.rbf_kernel import RBFKernel, postprocess_rbf

class RBFKernelDirectionalGrad(RBFKernel):
    r"""
    Computes a covariance matrix of the RBF kernel that models the covariance
    between the values and directional derivatives for inputs :math:`\mathbf{x_1}`
    and :math:`\mathbf{x_2}`.

    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    Args:
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
             batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`.
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`.
        :attr:`lengthscale_constraint` (Constraint, optional):
            Set this ifa you want to apply a constraint to the lengthscale parameter. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_shape` arguments.

    Example:
        >>>>  # generate training data
        >>>>  n   = 100
        >>>>  dim = 4
        >>>>  train_x = torch.rand(n,dim)
        >>>>  # number of inducing points
        >>>>  num_inducing = 20
        >>>>  # set directions
        >>>>  n_directions = 2
        >>>>  V = train_x[0:n_directions]      
        >>>>  k = RBFKernelDirectionalGrad()
        >>>>  # must set number of directions
        >>>>  k.set_n_dir(n_directions)      
        >>>>  params = {'V':V,'num_inducing':num_inducing}
        >>>>  K = k(train_x,diag=False,**params)
        >>>>  print(K.detach().numpy())

    """

    def forward(self, x1, x2, v1,v2,diag=False, **params):

        # TODO:
        # 1. pass in the directions for x1 and x2
        # 

        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]
        # directions
        n_dir1 = v1.shape[-2]
        n_dir2 = v2.shape[-2]

        # normalize directions
        v1 = (v1.T/torch.norm(v1,dim=1)).T
        v2 = (v2.T/torch.norm(v1,dim=1)).T


        # n1*(dim+1) x n2*(dim+1)
        K = torch.zeros(n1 * (n_dir1+ 1), n2 * (n_dir2+ 1), device=x1.device, dtype=x1.dtype)

        if not diag:
            # Scale the inputs by the lengthscale (for stability)
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)

            # project inputs onto directions
            x1_v1 = x1_ @ v1.T
            x1_v2 = x1_ @ v2.T
            x2_v1 = x2_ @ v1.T
            x2_v2 = x2_ @ v2.T

            # 1) Kernel block
            diff = self.covar_dist(x1_, x2_, square_dist=True, dist_postprocess_func=postprocess_rbf, **params)
            K_11 = diff
            K[:n1, :n2] = K_11

            # 2) First gradient block # (n1 x n2*n_dir2)
            # Form all possible rank-1 products for the gradient block
            outer = x1_v2.unsqueeze(1) - x2_v2
            # reshape to (n1 x n2*n_dir2) matrix
            outer = outer.view(n1,n2*n_dir2)
            # repeat_interleave row entries of the (n1 x n2) K_11 matrix  n_dir2 times
            K[:n1, n2:] = outer*K_11.repeat_interleave(n_dir2,1)


            # 3) Second gradient block # (n1*n_dir1 x n2)
            outer2 = x1_v1.unsqueeze(1) - x2_v1
            # reshape to (n1 x n2*n_dir2) matrix
            outer2 = outer.view(n1,n2*n_dir2)
            # repeat_interleave row entries of the (n1 x n2) K_11 matrix  n_dir2 times
            K[n1:, :n2] = -outer2 *K_11.repeat_interleave(n_dir2,1)


            # TODO:
            # use directions for both x1 and x2 to compute this block
            # 
            # 4) Hessian block # (n1*n_dir1 x n2In_dir2)
            
            K[n1:, n2:] =
            
            # Symmetrize for stability
            if n1 == n2 and torch.eq(x1, x2).all():
                K = 0.5 * (K.transpose(-1, -2) + K)

            return K

        else:
            if not (n1 == n2 and torch.eq(x1, x2).all()):
                raise RuntimeError("diag=True only works when x1 == x2")

            kernel_diag = super(RBFKernelDirectionalGrad, self).forward(x1, x2, diag=True)
            grad_diag = torch.ones(*batch_shape, n2, n_dir, device=x1.device, dtype=x1.dtype) / self.lengthscale.pow(2)
            grad_diag = grad_diag.transpose(-1, -2).reshape(*batch_shape, n2 * n_dir)
            k_diag = torch.cat((kernel_diag, grad_diag), dim=-1)
            return k_diag




if __name__ == '__main__':
  
  # generate training data
  n   = 100
  dim = 4
  train_x = torch.rand(n,dim)
  # number of inducing points
  num_inducing = 20
  # set directions
  n_directions = 4
  v1 = torch.eye(dim)[:n_directions]
  v2 = v1

  k = RBFKernelDirectionalGrad()
  # must set number of directions
  k.set_n_dir(n_directions)

  K = k(train_x,train_x,v1,v2,diag=False)
  print(K.detach().numpy())

  # if n_directions == dim
  # and directions are canonical basis
  # we can check answer with RBFKernelGrad
  # from gpytorch.kernels import RBFKernelGrad
  # kk = RBFKernelGrad()
  # KK = kk(train_x)
  # # print(KK.detach().numpy())
  # diff = KK.detach().numpy()-K.detach().numpy()
  # print(diff.max())
  # print(diff.min())
