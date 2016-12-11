include("./Linesearch.jl")
include("./Functions.jl")
include("./GradientDescent.jl")
include("./Evaluate.jl")
include("./Plotting.jl")

using Functions
using MAT
using GD
using Evaluate
using Plotting

dataLocation = "../data/mnist67.scale.1k.mat"

file = matopen(dataLocation)
tmp = read(file, "X")

X = full(tmp)
y = read(file,"y")
println("Loaded dataset")

rosen_x0 = zeros(2,1)

println(g_rosenbrock2([1;1]))

# optimize rosenbrock with backtracking linesearch and default pk
(x, status,vals, stops )= qn(rosen_x0,f_rosenbrock,g_rosenbrock2,1e-12,5000,20,20,1e-4,.9,"bt","gd")

plotArray(vals,"rosen-fval.svg")
plotArray(stops,"rosen-stop.svg")

#println(status)
#println(x)
#(x, status )= qn(rosen_x0,f_rosenbrock,g_rosenbrock,1e-12,1000,25,20,1e-4,.9,"bt","gd")
# (x, status )= qn(3000,f_square,g_square,1e-12,1000,20,20,1e-4,.9,"wolfe","gd")

x0 = zeros(size(X,2),1)

lambda = 0.1
iter = 100

println("Backtracking GradientDescent LogReg")
@time (x, status,vals,stops )= qn(x0,f_logreg(X,y,x0,lambda),g_logreg(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"bt","gd")
println(evaluate(X,y,x))

plotArray(vals,"gd-fval.svg")
plotArray(stops,"gd-stop.svg")

println("WolfeLS GradientDescent LogReg")
@time (x, status,vals,stops )== qn(x0,f_logreg(X,y,x0,lambda),g_logreg(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"wolfe","gd")
println(evaluate(X,y,x))

plotArray(vals,"wolfe-fval.svg")
plotArray(stops,"wolfe-stop.svg")

println("Wolfe BFGS LogReg")
@time (x, status,vals,stops )== qn(x0,f_logreg(X,y,x0,lambda),g_logreg(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"wolfe","bfgs")
println(evaluate(X,y,x))

plotArray(vals,"bfgs-fval.svg")
plotArray(stops,"bfgs-stop.svg")

println(status)

println(evaluate(X,y,x))
