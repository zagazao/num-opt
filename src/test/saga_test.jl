include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/SAGA.jl")
include("../data/data.jl")

using ProximalOperators

X, y = getOldData()

# init initial point x_0
x0 = zeros(size(X,2),1)

lambda = 0

iter = 1000
step =0.003

f = f_logreg(X,y,x0,lambda)
g = sub_g_logreg()

prox_operator = SqrNormL2(0)

@time (theta, strings, iter) = SAGA(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1),prox_operator)

grad = g_logreg(X,y,theta,lambda)(theta)
grad_norm = norm(grad, Inf)
@printf("Inf-Norm of gradient is %f.\n",grad_norm)
println(evaluate(X,y,theta,"logreg",false))

# Lippschitz = largest eigenvalue of Hessian*Hessian'
