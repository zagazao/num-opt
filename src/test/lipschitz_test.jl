include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/SAGA.jl")
include("../data/data.jl")


X, y = getOldData()


x0 = zeros(size(X,2),1)

H = logreg_hessian(X,y,x0)

(vec,val) = eig(H)
println(vec)
println(val)
