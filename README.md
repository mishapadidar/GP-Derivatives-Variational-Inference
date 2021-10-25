# GP-Derivatives-Variational-Inference

This repo contains code for the NeurIPS paper, Scaling Gaussian Processes with Derivatives Using Variational Inference.

All of our code leverages the GPyTorch framework for efficient computations and GPU acceleration. Much of the functionality of this code base is, or soon will be, built into GPyTorch.

The `graphite_environment.yml` can be used to create a compatible conda environment.

The directory structure is as follows:
- `directional_vi` the main methods used in the paper.
- `tests` contains basic tests showing how to use the methods.
-  `experiments` contains code for the experiments run in the paper, including the graph convolutional network, stellarator regression, rover, bunny, and UCI experiments. For experimental data please contact the authors.

For a basic introduction on how to use the methods from the paper visit the `tests` directory.
