include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/pgd.jl")
include("../data/data.jl")
include("../plot/Plotting.jl")

#using Plotting

X, y = getOldData()

x0 = zeros(size(X,2),1)

lambda = 0

iter = 1000
step = 0.01

f = f_logreg(X,y,x0,lambda)
g = g_logreg(X,y,x0,0)

#f = f_svm(X,y,x0,lambda)
#g = sub_g_svm()

prox_operator = SqrNormL2(lambda/2)

@time (theta,state,val_array,stop_array) = pgd(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1),prox_operator)

#plotArray(val_array, "sgd1.svg")

grad = g(theta)
grad_norm = norm(grad, Inf)
@printf("Inf-Norm of gradient is %f.\n",grad_norm)
println(evaluate(X,y,theta,"logreg",false))
