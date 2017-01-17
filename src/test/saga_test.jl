include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/SAGA.jl")
include("../data/data.jl")

using ProximalOperators

X, y = getFullData()

# init initial point x_0
x0 = zeros(size(X,2),1)
convexity = 0.0001
iter = 5000
lipschitz = 3
opt_step = 1 / 3 * ( convexity * iter + lipschitz)
step = 0.0001
# unregularized hinge-loss
f = f_svm(X,y,x0,0.1)
g = sub_g_x_svm()


#f = f_logreg(X,y,x0,0.1)
#g = sub_g_x_logreg()

prox_operator = NormL2(0.1)
(theta, strings, iter) = SAGA(X,y,x0,f,g,1e-8,0,iter,step,size(X,1),prox_operator)
grad = g_logreg(X,y,theta,0.1)(theta)
grad_norm = norm(grad, Inf)
@printf("Inf-Norm of gradient is %f.\n",grad_norm)

println(evaluate(X,y,theta,"svm",false))

# Lippschitz = largest eigenvalue of Hessian*Hessian'
