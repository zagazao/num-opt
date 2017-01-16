include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/SAGA.jl")
include("../data/data.jl")

using ProximalOperators

X, y = getFullData()

# init initial point x_0
x0 = zeros(size(X,2),1)
iter = 3000
step = 0.01
#f = f_svm(X,y,x0,0)
#g = sub_g_x_svm()
# unregularized logistic-regression
f = f_logreg(X,y,x0,0)
g = sub_g_x_logreg()

prox_operator = NormL2(0.1)
(theta, strings, iter) = SAGA(X,y,x0,f,g,1e-8,0,iter,step,size(X,1),prox_operator)
println(evaluate(X,y,theta,"logreg",false))

# Lippschitz = largest eigenvalue of Hessian*Hessian'
lippschitz = 0.5
convexity = 0.5
