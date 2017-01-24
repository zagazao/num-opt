include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/pgd.jl")
include("../opt/agd.jl")
include("../opt/GD-const.jl")
include("../data/data.jl")
include("../plot/Plotting.jl")

using Plotting

X, y = getOldData()

x0 = zeros(size(X,2),1)

lambda = 0.000001

iter = 50
step = 0.01

f = f_logreg(X,y,x0,lambda)
g = g_logreg(X,y,x0,0)


g_l2 = g_logreg(X,y,x0,lambda)

prox_operator = SqrNormL2(lambda/2)

@time (x_pgd,state,val_array_pgd,stop_array_pgd) = pgd(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1),prox_operator)
@time (x_agd,state,val_array_agd,stop_array_agd) = agd(X,y,x0,f,g_l2,1e-8,lambda,iter,step,size(X,1),prox_operator)
@time (x_gd,state,val_array_gd,stop_array_gd) = gd(X,y,x0,f,g_l2,1e-8,lambda,iter,step,size(X,1))

plotTripleArray(val_array_pgd,val_array_agd,val_array_gd, "GD-FVAL-COMPARE.svg")
plotTripleArray(stop_array_pgd,stop_array_agd,stop_array_gd, "GD-STOP-COMPARE.svg")
