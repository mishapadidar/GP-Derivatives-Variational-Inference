#!/usr/bin/env python3
import torch

from gpytorch.lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from gpytorch.kernels.rbf_kernel import RBFKernel, postprocess_rbf


class RBFKernelDirectionalGrad(RBFKernel):
    r"""
    Pass in v1 and v2 through the params. If v1 has n_dir1 directions per
    point in x2 then it should be shape n1*n_dir1 x dim. The directions
    are assumed to be stored in blocks so that the first n_dir1 directions
    belong to x1[0] and the second n_dir1 directions belong to x1[1] etc.
    
    If you have a single set of global directions such as torch.eye(dim), then 
    you can repeat those to make v1 and v2 with 
    v1 = torch.eye(dim).repeat(n1,1)

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
            Set this if you want to apply a constraint to the lengthscale parameter. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_shape` arguments.


    """

    def forward(self, x1, x2, diag=False, **params):
        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]


        v1 = params['v1']
        v2 = params['v2']
        # number of directions per point
        n_dir1 = int(v1.shape[-2]/n1)
        n_dir2 = int(v2.shape[-2]/n2)
        assert n_dir1 == n_dir2, "v1 and v2 must contain same number of directions"

        self.set_num_directions(n_dir1)
        # normalize directions
        v1 = (v1.T/torch.norm(v1,dim=1)).T
        v2 = (v2.T/torch.norm(v2,dim=1)).T

        # K = torch.zeros(*batch_shape, n1 * (d + 1), n2 * (d + 1), device=x1.device, dtype=x1.dtype)
        K = torch.zeros(*batch_shape, n1 * (n_dir1 + 1), n2 * (n_dir2 + 1), device=x1.device, dtype=x1.dtype)
        K = torch.zeros(*batch_shape, n1 * (n_dir1 + 1), n2 * (n_dir2 + 1), device=x1.device, dtype=x1.dtype)

  
        if not diag:
            # Scale the inputs by the lengthscale (for stability)
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)

            # 1) Kernel block
            diff = self.covar_dist(x1_, x2_, square_dist=True, dist_postprocess_func=postprocess_rbf, **params)
            K_11 = diff
            K[..., :n1, :n2] = K_11


            # 2) First gradient block
            x2_v2 = x2_.reshape(n2,1,d).bmm(torch.transpose(v2.reshape(n2,n_dir2,d),-2,-1))
            x1_v2 = x1_ @ v2.T
            outer  = x1_v2 - x2_v2.flatten()
            # permute cols so we get blocks for v1,v2,v3,...
            pi2 = torch.arange(n2 * (n_dir2)).view(n2,n_dir2).t().reshape((n2 * (n_dir2)))
            outer1 = outer[:,pi2]/ self.lengthscale.unsqueeze(-2)
            K[..., :n1, n2:] = outer1 * K_11.repeat([*([1] * (n_batch_dims + 1)), n_dir2]) 

            # Second gradient block
            x1_v1 = x1_.reshape(n1,1,d).bmm(torch.transpose(v1.reshape(n1,n_dir1,d),-2,-1))
            x2_v1 = x2_ @ v1.T
            outer  = x1_v1.flatten() - x2_v1
            # permute cols so we get blocks for v1,v2,v3,...
            pi2 = torch.arange(n1 * (n_dir1)).view(n1,n_dir1).t().reshape((n1 * (n_dir1)))
            outer2 = outer[:,pi2]
            outer2  = outer2.t() / self.lengthscale.unsqueeze(-2)
            K[..., n1:, :n2] = -outer2 * K_11.repeat([n_dir1,*([1] * (n_batch_dims + 1))]) 


            # 4) Hessian block (n1*n_dir1, n2*n_dir2)
            outer3 = outer1.repeat(1, n_dir2, 1) * outer2.repeat(1,1,n_dir1)  
            # kp = KroneckerProductLazyTensor(
            #     torch.eye(n_dir1,n_dir2, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1) / self.lengthscale.pow(2),
            #     torch.ones(n1, n2, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1),
            # )
            # chain_rule = kp.evaluate() - outer3

            # mishas rules
            kp = v1 @ v2.T / self.lengthscale.pow(2)
            kp = kp[:,pi2]
            kp = kp[pi2,:]
            # print(kp)
            chain_rule = kp - outer3

            K[..., n1:, n2:] = chain_rule * K_11.repeat([*([1] * n_batch_dims), n_dir1,n_dir2])
            
            # # print(outer3)
            # # print(chain_rule)
            # ii = 0
            # jj = 2
            # ii_v = 0
            # jj_v = 5

            # # directional grad hessian block v^TH_{ij}v
            # aa = (torch.eye(2)/self.lengthscale.unsqueeze(-2))**2
            # aa = aa-torch.outer(x1_[ii]-x2_[jj],x1_[ii]-x2_[jj])/(self.lengthscale.unsqueeze(-2))**2
            # aa = aa*K_11[ii,jj]
            # aa = v1[ii_v] @ aa @ v2[jj_v]
            # # print(aa)

            # # # regular hessian block H_{ij}
            # # aa = (torch.eye(2)/self.lengthscale.unsqueeze(-2))**2
            # # aa = aa-torch.outer(x1_[ii]-x2_[jj],x1_[ii]-x2_[jj])/(self.lengthscale.unsqueeze(-2))**2
            # # aa = aa*K_11[ii,jj]
            # # print(aa)

            # # outer1 computation
            # aa = (x1_[ii]-x2_[jj])/(self.lengthscale.unsqueeze(-2))
            # aa1 = aa @ v2[jj_v]
            # print(aa1)
            # print(outer1)
            # # outer2 computation
            # aa = (x1_[ii]-x2_[jj])/(self.lengthscale.unsqueeze(-2))
            # aa2 = aa @ v1[ii_v]
            # print(aa2)
            # print(outer2)

            # # directional grad hessian block in terms of outer1 and outer2
            # aa = (v2[jj_v]@v1[ii_v]/(self.lengthscale**2) -  aa1*aa2)*K_11[ii,jj]
            # print(aa)

            # [] repeat outer1 by stacking vertically
            # [] repeat outer2 by stacking horizontally (side by side)
            # [] kronecker matrix is kp = v2v1^T or something??? idk
            # [] final formula 
            #    (kp/l^2 - outer1*outer2)*K.repeat...

            # =================================================

            # # New attempt at first gradient block
            # outer = x1_.view(*batch_shape, n1, 1, d) - x2_.view(*batch_shape, 1, n2, d)
            # outer = outer / self.lengthscale.unsqueeze(-2)
            # outer = torch.transpose(outer, -1, -2).contiguous()
            # outer1 = outer.view(*batch_shape, n1, n2 * d)
            # Grad_block = outer1 * K_11.repeat([*([1] * (n_batch_dims + 1)), d])
            # # Apply a perfect shuffle permutation to match the MutiTask ordering
            # pi1 = torch.arange(n1).view(1, n1).t().reshape((n1))
            # pi2 = torch.arange(n2 * (d)).view(d, n2).t().reshape((n2 * (d)))
            # Grad_block = Grad_block[pi1, :][:, pi2]
            # for ii in range(n_dir2):
            #   vv = v2[ii::n_dir2].flatten()
            #   R = Grad_block*vv
            #   R = R.reshape(n1,n2,d).sum(dim = 2)
            #   K[..., :n1, n2+ii*n2:n2+(ii+1)*n2] = R


            # # New attempt at second gradient block
            # outer2 = outer.transpose(-1, -3).reshape(*batch_shape, n2, n1 * d)
            # outer2 = outer2.transpose(-1, -2)
            # Grad_block = -outer2 * K_11.repeat([*([1] * n_batch_dims), d, 1])
            # # Apply a perfect shuffle permutation to match the MutiTask ordering
            # pi1 = torch.arange(n1 * (d)).view(d, n1).t().reshape((n1 * (d)))
            # pi2 = torch.arange(n2).view(1, n2).t().reshape(n2)
            # Grad_block = Grad_block[pi1, :][:, pi2]
            # for ii in range(n_dir1):
            #   vv = v1[ii::n_dir1].flatten()
            #   R = Grad_block.T*vv
            #   R = R.reshape(n2,n1,d).sum(dim=2).T
            #   K[...,  n1+ii*n1:n1+(ii+1)*n1,:n2] = R


            # # new hessian block
            # outer3 = outer1.repeat([*([1] * n_batch_dims), d, 1]) * outer2.repeat([*([1] * (n_batch_dims + 1)), d])
            # kp = KroneckerProductLazyTensor(
            #     torch.eye(d, d, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1) / self.lengthscale.pow(2),
            #     torch.ones(n1, n2, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1),
            # )
            # chain_rule = kp.evaluate() - outer3
            # Hess_block = chain_rule * K_11.repeat([*([1] * n_batch_dims), d, d])
            # # Apply a perfect shuffle permutation to match the MutiTask ordering
            # pi1 = torch.arange(n1 * (d)).view(d, n1).t().reshape((n1 * (d)))
            # pi2 = torch.arange(n2*d).view(d, n2).t().reshape(d*n2)
            # Hess_block = Hess_block[pi1, :][:, pi2]
            # for ii in range(n_dir1):
            #   for jj in range(n_dir2):
            #     vv1 = v1[ii::n_dir1].flatten()
            #     vv2 = v2[jj::n_dir2].flatten()
            #     # multiply by directions
            #     HH = ((Hess_block * vv2.flatten()).T * vv1.flatten()).T
            #     # reshape to separate blocks from each other
            #     HH = HH.view(n1,d,n2*d).permute(0,-1,1).view(n1,n2,d,d).transpose(-1,-2)
            #     # take sums
            #     K[...,  n1+ii*n1:n1+(ii+1)*n1,n2+jj*n2:n2+(jj+1)*n2] = HH.sum(dim = (-1,-2))



            # print(K[..., n1:, n2:]/K_11.repeat([*([1] * n_batch_dims), n_dir1,n_dir2]))
            # print(Hess_block)
            # print(Kappa[..., n1:, n2:] - K[..., n1:, n2:])
            # print(Kappa[..., n1:, n2:])
            # print(K[..., n1:, n2:])
            # ii = 0
            # jj = 1
            # aa = (torch.eye(2)/self.lengthscale.unsqueeze(-2))**2
            # aa = aa-torch.outer(x1_[ii]-x2_[jj],x1_[ii]-x2_[jj])/(self.lengthscale.unsqueeze(-2))**2
            # # aa = aa*K_11[ii,jj]
            # aa = v1[0] @ aa @ v2[1]
            # print(v1[0]@ torch.outer(x1_[ii]-x2_[jj],x1_[ii]-x2_[jj])/(self.lengthscale.unsqueeze(-2))**2 @ v2[3])
            # print(aa)
            # quit()


            # Apply a perfect shuffle permutation to match the MutiTask ordering
            pi1 = torch.arange(n1 * (n_dir1 + 1)).view(n_dir1 + 1, n1).t().reshape((n1 * (n_dir1 + 1)))
            pi2 = torch.arange(n2 * (n_dir2 + 1)).view(n_dir2 + 1, n2).t().reshape((n2 * (n_dir2 + 1)))
            K = K[..., pi1, :][..., :, pi2]
            return K

        else:
            if not (n1 == n2 and torch.eq(x1, x2).all()):
                raise RuntimeError("diag=True only works when x1 == x2")

            kernel_diag = super(RBFKernelGrad, self).forward(x1, x2, diag=True)
            grad_diag = torch.ones(*batch_shape, n2, d, device=x1.device, dtype=x1.dtype) / self.lengthscale.pow(2)
            grad_diag = grad_diag.transpose(-1, -2).reshape(*batch_shape, n2 * d)
            k_diag = torch.cat((kernel_diag, grad_diag), dim=-1)
            pi = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().reshape((n2 * (d + 1)))
            return k_diag[..., pi]

    def set_num_directions(self,num_directions):
        self.n_dir1 = num_directions

    def num_outputs_per_input(self, x1, x2):
        return self.n_dir1 +1



if __name__ == '__main__':

  torch.manual_seed(0)
  # generate training data
  n1   = 100
  n2   = n1
  dim = 2
  train_x  = torch.rand(n1,dim)
  # train_x2 = torch.rand(n2,dim)
  train_x2 = train_x
  # set directions
  n_directions = 2
  # v1 = torch.eye(dim)[:n_directions]
  v1 = torch.rand(n_directions,dim)
  v1 = v1.repeat(n1,1)
  # v2 = torch.eye(dim)[:n_directions]
  # v2 = torch.rand(n_directions,dim)
  # v2 = v2.repeat(n2,1)
  v2 = v1
  v1 = (v1.T/torch.norm(v1,dim=1)).T
  v2 = (v2.T/torch.norm(v2,dim=1)).T

  k = RBFKernelDirectionalGrad()
  params = {'v1':v1,'v2':v2}
  K = k(train_x,train_x2, **params)
  print(K.detach().numpy().shape)

  torch.cholesky(K.add_jitter().evaluate())
  # verify against RBFKernelGrad
  # from gpytorch.kernels import RBFKernelGrad
  # kk = RBFKernelGrad()
  # KK = kk(train_x,train_x2)
  # print(KK.detach().numpy() - K.detach().numpy())
