include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/sgd.jl")
include("../data/data.jl")

X, y = getOldData()

x0 = zeros(size(X,2),1)

iter = 40
step = 0.01

f = f_logreg(X,y,x0,0)
g = sub_g_logreg()

@time (theta, strings, iter) = sgd(X,y,x0,f,g,1e-8,0,iter,step,size(X,1))

grad = g_logreg(X,y,theta,0)(theta)
grad_norm = norm(grad, Inf)
@printf("Inf-Norm of gradient is %f.\n",grad_norm)
println(evaluate(X,y,theta,"logreg",false))
