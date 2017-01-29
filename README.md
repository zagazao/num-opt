# num-opt

Implementation of some optimization algorithms for the course Numerical Optimization.

You can run a comparison of some stochastic algorithms via

```
julia src/test/main.jl --lam 0.001 --step 0.001 --iter 1000 --mode logreg
```
, where lam is the regularization value, step the step size, iter the number of iterations and mode can be logreg or svm, depending on which function should be minimized.

Afterwards some plots will be created inside the directroy, comparing SGD, SAG, SAGA (with Prox).
