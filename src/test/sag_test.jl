include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/SAG.jl")
include("../data/data.jl")

X, y = getFullData()

# init initial point x_0
x0 = zeros(size(X,2),1)

lambda = 5

iter = 250
step =0.01

f = f_logreg(X,y,x0,lambda)
g = sub_g_logreg()

#f = f_svm(X,y,x0,lambda)
#g = sub_g_svm()


@time (theta, strings, iter) = SAG(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1))

grad = g_logreg(X,y,theta,lambda)(theta)
grad_norm = norm(grad, Inf)
@printf("Inf-Norm of gradient is %f.\n",grad_norm)
println(evaluate(X,y,theta,"logreg",false))

# Lippschitz = largest eigenvalue of Hessian*Hessian'
