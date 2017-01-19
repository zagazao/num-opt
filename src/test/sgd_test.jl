include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/sgd.jl")
include("../data/data.jl")
include("../plot/Plotting.jl")

using Plotting

X, y = getOldData()

x0 = zeros(size(X,2),1)

lambda = 0.1

iter = 15000
step = 0.0001

#f = f_logreg(X,y,x0,lambda)
#g = sub_g_logreg()

f = f_svm(X,y,x0,lambda)
g = sub_g_svm()

@time (theta,state,val_array,stop_array) = sgd(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1))

plotArray(val_array, "sgd1.svg")

grad = g_logreg(X,y,theta,lambda)(theta)
grad_norm = norm(grad, Inf)
@printf("Inf-Norm of gradient is %f.\n",grad_norm)
println(evaluate(X,y,theta,"svm",false))
