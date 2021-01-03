
# About
Bayesian optimization is used in many optimization problems and it is always the optimal choice for the global optimization of  with expensive evaluations.
But it is difficult to find an opencl implementation of Bayesian optimization.
I made a simple implementation with opencl in this project.

# Features & Disadvantages
- blas and linear algebra level optimization
- Performance gains occur only when significant iteration is required.
- Strong assumptions were used and not sufficiently verified.( about pseudo inverse matrix)
- It takes a lot of overhead to generate random numbers.

# Warning
- For performance purposes, use a library like [clblast](https://github.com/CNugteren/CLBlast) or [arrayfire](https://github.com/arrayfire/arrayfire).
# Requirement
- opencl 
- C++>=14
- cmake

# Report
[report](report/project.pdf)

# Reference 
[GP-Bayesian-Optimizer](https://github.com/TheisFerre/GP-Bayesian-Optimizer)

# Contribute
Always welcome.