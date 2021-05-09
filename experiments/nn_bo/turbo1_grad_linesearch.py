###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from turbo_utils import from_unit_cube, latin_hypercube, to_unit_cube


class Turbo1Grad:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        turbo1.optimize()  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        train_gp,
        sample_from_gp,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    ):

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int) 
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # train_gp function handle
        self.train_gp = train_gp
        self.sample_from_gp = sample_from_gp

        # Save function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_improv = 3
        assert batch_size > self.n_improv
        self.n_cand = min(100 * self.dim, 5000)
        #self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.failtol = 5 # MP
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6 
        self.length_init = 0.8

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, self.dim + 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        if np.min(fX_next) < np.min(self._fX[:,0]) - 1e-3 * math.fabs(np.min(self._fX[:,0])):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= 2.0
            self.failcount = 0

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        #assert X.min() >= 0.0 and X.max() <= 1.0   # MP: turned off b/c model improv

        # Standardize function values.
        mu, sigma = np.median(fX, axis=0)[0], fX.std(axis=0)[0]
        fX[:,0] = (deepcopy(fX[:,0]) - mu) / sigma
        # Standardize gradients
        fX[:,1:] = deepcopy(fX[:,1:]) / sigma
        # do from_unit_cube mapping on gradients (b/c X got mapped to unit cube)
        fX[:,1:] = deepcopy(fX[:,1:]) * (self.ub-self.lb)

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp,likelihood = self.train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX[:, 0].argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            # predict function values (self.n_cand, self.batch_size)
            y_cand = self.sample_from_gp(gp,likelihood,X_cand_torch,self.batch_size).cpu().detach().numpy()

            #mu_th = torch.from_numpy(mu).to(device=y_cand.device, dtype=y_cand.dtype)
            #sigma_th = torch.from_numpy(sigma).to(device=y_cand.device, dtype=y_cand.dtype)

            #y_cand = mu_th.view(self.dim + 1, 1, 1) + sigma_th.view(self.dim + 1, 1, 1) * y_cand
            #y_cand = y_cand.cpu().detach().numpy()

        # Remove the torch variables
        #del X_torch, y_torch, X_cand_torch, gp, mu_th, sigma_th
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled function values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, x_center

    #def _select_candidates(self, X_cand, y_cand):
    #    """Select candidates."""
    #    X_next = np.ones((self.batch_size, self.dim))
    #    for i in range(self.batch_size):
    #        # Pick the best point and make sure we never pick it again
    #        indbest = np.argmin(y_cand[0, :, i])
    #        X_next[i, :] = deepcopy(X_cand[indbest, :])
    #        y_cand[0, indbest, :] = np.inf
    #    return X_next
    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        y_next = np.ones(self.batch_size)
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_next[i] = y_cand[indbest,i]
            y_cand[indbest, :] = np.inf
        return X_next,y_next

    def _create_candidates_linesearch(self, X, fX, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0   # MP: turned off b/c model improv

        # destandardize X
        X = from_unit_cube(X,self.lb,self.ub)
        # find TR center 
        x_center = X[fX[:, 0].argmin().item(), :][None, :]
        g_center = fX[fX[:, 0].argmin().item(), 1:][None, :] # gradient
        # max learning rate to stay within cube boundary
        b  = -(self.ub-x_center)/g_center
        c  = -(self.lb-x_center)/g_center
        d  = np.hstack((b,c)).flatten()
        alpha_max= np.min([dj for dj in d if dj>=0])
        print(alpha_max)
        # generate line search points
        betas = (2.0**np.arange(-22,4,1)) # 1e-8 to 16
        betas = betas[betas < alpha_max]
        if len(betas) > 0:
          X_temp = x_center.flatten()-np.array([ beta*g_center.flatten() for beta in betas])
          fs  = [self.f(xx)[0] for xx in X_temp]
          print(np.argmin(fs),np.min(fs))
          # map points to unit cube
          X_temp = to_unit_cube(X_temp,self.lb,self.ub)

        # restandardize X
        X = to_unit_cube(X,self.lb,self.ub)

        ## Standardize function values.
        mu, sigma = np.median(fX, axis=0)[0], fX.std(axis=0)[0]
        fX[:,0] = (deepcopy(fX[:,0]) - mu) / sigma
        # Standardize gradients
        fX[:,1:] = deepcopy(fX[:,1:]) / sigma
        # do from_unit_cube mapping on gradients (b/c X got mapped to unit cube)
        fX[:,1:] = deepcopy(fX[:,1:]) * (self.ub-self.lb)

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp,likelihood = self.train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX[:, 0].argmin().item(), :][None, :]
        #g_center = fX[fX[:, 0].argmin().item(), 1:][None, :] # gradient
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # append linesearch points
        if self.failcount < 1 and len(betas) > 0:
          X_cand = np.vstack((X_cand,X_temp))
        else: # just use TS
          pass

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            # predict function values (self.n_cand, self.batch_size)
            y_cand = self.sample_from_gp(gp,likelihood,X_cand_torch,self.batch_size).cpu().detach().numpy()

        # Remove the torch variables
        #del X_torch, y_torch, X_cand_torch, gp, mu_th, sigma_th
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled function values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, x_center



    def _model_improvement(self, x_center,X_next):
        """improve the model
        generate n_improv points that are affinely independent
        to the first (batch_size - n_improv) points in X_next
        """
        x_center = x_center.flatten() # it is 2d by default
        # shift X_next
        X_new = X_next.copy() - x_center
        # take the first batch_size - n_improv points
        X_new = X_new[:self.batch_size - self.n_improv]
        # generate random points
        U = np.random.randn(2*self.n_improv,self.dim) # 2*n_improv points
        A = np.vstack((X_new,U))
        # rescale for conditioning
        A = A.T/np.linalg.norm(A,axis=1)
        # get orthogonal vectors
        Q,_ = np.linalg.qr(A)
        # take the last n_improv points
        X_new = (Q.T)[-self.n_improv:]
        # upper bound the length so that the points stay within the TR
        scales = (self.length/4)*np.ones(self.n_improv)

        # makes sure points are within unit cube
        # suffers because x_center is often on boundary of unit cube
        # making the step size = 0
        #for ii in range(self.n_improv):
        #  b  = -x_center/X_new[ii]
        #  c  = (1 - x_center)/X_new[ii]
        #  d  = np.hstack((b,c,self.length))
        #  scales[ii] = np.min([dj for dj in d if dj>=0])

        # scale them by the TR size and shift to center
        X_new = np.diag(scales) @ X_new + x_center
        # make improvement points the last evaluation points
        X_next[-self.n_improv:] = X_new.copy()
        return X_next

    def optimize(self):
        """Run the full optimization process."""
        while self.n_evals < self.max_evals:
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX[:, 0].min()
                print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            # Initialize parameters
            self._restart()

            # Generate and evalute initial design points
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            fX_init = np.vstack([self.f(x) for x in X_init])

            # Update budget and set as initial data for this TR
            self.n_evals += self.n_init
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            if self.verbose:
                fbest = self._fX[:, 0].min()
                print(f"Starting from fbest = {fbest:.4}")
                sys.stdout.flush()

            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self._fX)

                # Create th next batch
                #X_cand, y_cand, x_center = self._create_candidates(
                #    X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                #)
                X_cand, y_cand,x_center = self._create_candidates_linesearch(
                    X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next,y_next = self._select_candidates(X_cand, y_cand)

                # make sure we have model improvement points
                #X_next = self._model_improvement(x_center,X_next)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)

                # Evaluate batch
                fX_next = np.vstack([self.f(x) for x in X_next])
                #print(fX_next[:,0] - y_next)

                # Update trust region (based only on function values)
                self._adjust_length(fX_next[:,0])

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if self.verbose and fX_next[:, 0].min() < self.fX[:, 0].min():
                    n_evals, fbest = self.n_evals, fX_next[:, 0].min()
                    print(f"{n_evals}) New best: {fbest:.4}")
                    sys.stdout.flush()

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))


