
## Basic tests to show usage and functionality of methods
To run the code in this directory use `python3 filename`. 

- `test_dsvgp.py` can be used to test a variational GP with directional derivatives, i.e. DSVGP or DPPGPR
- `test_dfree_dsvgp.py` can be used to train DSVGP on a data set that has no derivative information.
- `test_grad_svgp.py` runs a multi-output stochastic variational GP with full derivative information. 
- `test_traditional_vi.py` runs SVGP or PPGPR.

