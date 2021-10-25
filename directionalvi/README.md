## Main Methods for Variational GP computions
This directory contains the main components of the methods for a variational GP with directional derivatives, as well as GPs with derivatives.
Much of this functionality is, or soon will be, incorporated into GPyTorch.

The RBF directional derivative kernel is `RBFKernelDirectionalGrad.py`.

The following files are the main scripts to initialize and run a method. To run an instance of DSVGP you would only need to import `directional_vi.py`. See the tests directory for usability.
- `directional_vi.py` contains the methods for initializing and running a variational GP with directional derivatives.
- `shared_directional_vi.py` contains the main methods for running a variational GP with directional derivatives with shared inducing directions.
- `dfree_directional_vi.py` contains the main methods for running a variational GP with directional derivatives on a dataset that does not have any derivative information.
- `traditional_vi.py` runs a standard SVGP.
- `grad_svgp.py` runs a multi-output SVGP with full derivative information.

Variational Stategies are used for prediction in Variational GPs.
- `DirectionalGradVariationalStrategy.py` is the workhorse variational strategy for GPs with directional derivatives.
- `DFreeDirectionalGradVariationalStrategy.py` allows DSVGP and DPPGPR to train on data without derivative labels.
- `SharedDirectionalGradVariationalStrategy.py` allows DSVGP and DPPGPR to use shared inducing directions.
- `CiqDirectionalGradVariationalStrategy.py` allows DSVGP and DPPGPR to leverage contour integral quadrature.
- `GradVariationalStrategy.py` is the variational strategy for a stochastic variational gaussian process with full derivative information.


The `utils` directory contains useful helper functions.
