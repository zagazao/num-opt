include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/SAGA.jl")
include("../opt/sgd.jl")
include("../data/data.jl")
include("../plot/Plotting.jl")

using ProximalOperators
using Plotting

X, y = getFullData()

# init initial point x_0
x0 = zeros(size(X,2),1)

lambda = 0.1

iter = 300
step =0.005


#f = f_logreg(X,y,x0,lambda)
#g = sub_g_logreg()

f = f_svm(X,y,x0,lambda)
g = sub_g_svm()

prox_operator = SqrNormL2(lambda)

@time (theta, strings, iter,val_array_saga) = SAGA(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1),prox_operator)

println(evaluate(X,y,theta,"svm",false))

@time (theta,state,val_array,stop_array) = sgd(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1))

plotDoubleArray(val_array_saga,val_array, "saga1.svg")

grad = g_logreg(X,y,theta,lambda)(theta)
grad_norm = norm(grad, Inf)
@printf("Inf-Norm of gradient is %f.\n",grad_norm)
println(evaluate(X,y,theta,"svm",false))
# Lippschitz = largest eigenvalue of Hessian*Hessian'
