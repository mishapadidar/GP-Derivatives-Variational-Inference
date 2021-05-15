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
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
from torch.quasirandom import SobolEngine

from gp import train_gp
from turbo_utils import from_unit_cube, latin_hypercube, to_unit_cube


class myBO:
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
        batch_size=1,
        verbose=True,
        use_ard=True,
        n_training_steps=50,
        max_cholesky_size=2000,
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
        assert dtype == "float32" or dtype == "float64" or dtype == "double"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

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
        # self.n_cand = min(100 * self.dim, 5000)
        self.n_cand = batch_size
        self.n_evals = 0
        # self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        # self.dtype = torch.float32 if dtype == "float32" else torch.float64
        if dtype == "float32":
          self.dtype = torch.float32
        elif dtype == "float64": 
          self.dtype = torch.float64
        elif dtype == "double":
          self.dtype = torch.double

        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

 

    def _create_candidates(self, X, fX, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), self.dtype
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        
        # optimize acquisition function to get X_cand
        def lower_confidence_bound(x, kappa=2):
            preds = gp(x.reshape(-1,self.dim))
            # mean = torch.tensor([0.])
            # sigma = torch.tensor([0.])
            # mean = torch.cat([mean, preds.mean.cpu()])
            # sigma = torch.cat([sigma, preds.variance.sqrt()])
            mean = preds.mean.cpu()
            sigma = preds.variance.sqrt().cpu()
            # print("evaluate lcb", (mean - kappa * sigma).item())
            return mean - kappa * sigma

        def find_a_candidate(x_init, lower_bound=0, upper_bound=1):
            assert lower_bound == 0 and upper_bound == 1
            # transform x to an unconstrained domain
            constraint = constraints.interval(lower_bound, upper_bound)
            unconstrained_x_init = transform_to(constraint).inv(x_init)
            unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
            minimizer = optim.LBFGS([unconstrained_x], line_search_fn='strong_wolfe')
            # print("Trying to find a candidate")
            
            def closure():
                minimizer.zero_grad()
                x = transform_to(constraint)(unconstrained_x)
                y = lower_confidence_bound(x)
                autograd.backward(x, autograd.grad(y, x))
                return y
            minimizer.step(closure)
            
            # after finding a candidate in the unconstrained domain,
            # convert it back to original domain.
            x = transform_to(constraint)(unconstrained_x)
            # print("candidate found!")
            return x.detach()

        # collect X_cand
        X_cand = np.ones((self.n_cand, self.dim))
        X_init = latin_hypercube(self.n_cand, self.dim)
        X_init = torch.tensor(X_init)
        X_init = X_init.to(dtype=dtype, device=device)
        for i in range(self.n_cand):
            X_cand[i,:] = find_a_candidate(X_init[i, :]).cpu()
        
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
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf
        return X_next

    def optimize(self):
        """Run the full optimization process."""
        while self.n_evals < self.max_evals:
            
            # Generate and evalute initial design points
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            fX_init = np.array([[self.f(x)] for x in X_init])
            print("Finish initial sampling")
            # Update budget and set as initial data for this TR
            self.n_evals += self.n_init
            
            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            
            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals:
                # Warp inputs
                X = to_unit_cube(deepcopy(self.X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self.fX).ravel()

                # Create th next batch
                X_cand, y_cand, _ = self._create_candidates(
                    X, fX, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next = self._select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)

                # Evaluate batch
                fX_next = np.array([[self.f(x)] for x in X_next])

                # Update budget and append data
                self.n_evals += self.batch_size
                
                if self.verbose:
                    n_evals, fbest = self.n_evals, fX_next.min()
                    print(f"n_evals: {n_evals}, New eval: {fbest:.4}")
                    sys.stdout.flush()

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))