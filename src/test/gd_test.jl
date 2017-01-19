include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/GradientDescent.jl")
include("../data/data.jl")


X, y = getOldData()

lambda = 0.1
iter = 100

x0 = zeros(size(X,2),1)

#f = f_logreg(X,y,x0,lambda)
#g = g_logreg(X,y,x0,lambda)


f = f_svm(X,y,x0,lambda)
g = sub_g_svm()

@time (theta, status,vals1,stops1 ) = qn(x0,f,g,1e-12,iter,20,20,1e-4,.9,"bt","gd")

grad = g(theta)
grad_norm = norm(grad, Inf)
@printf("Inf-Norm of gradient is %f.\n",grad_norm)
println(evaluate(X,y,theta,"logreg",false))
