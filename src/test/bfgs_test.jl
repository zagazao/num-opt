include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/GradientDescent.jl")
include("../data/data.jl")

X, y = getOldData()

lambda = 10
iter = 100

x0 = zeros(size(X,2),1)

f = f_logreg(X,y,x0,lambda)
g = g_logreg(X,y,x0,lambda)

@time (x, status,vals1,stops1 ) = qn(x0,f_logreg(X,y,x0,lambda),g_logreg(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"wolfe","bfgs")
