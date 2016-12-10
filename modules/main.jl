include("./Linesearch.jl")
include("./Functions.jl")
include("./GradientDescent.jl")
include("./Evaluate.jl")
#include("./Plotting.jl")

using Functions
using MAT
using GD
using Evaluate
#using Plotting

@time println("hello world")

dataLocation = "../data/mnist67.scale.1k.mat"

file = matopen(dataLocation)
tmp = read(file, "X")

X = full(tmp)
y = read(file,"y")
println("Loaded dataset")

# optimize rosenbrock with backtracking linesearch and default pk
#(x, status )= qn([2 ;1],f_rosenbrock,g_rosenbrock,1e-12,100,20,20,1e-4,.9,"wolfe","gd")
#(x, status )= qn([2 ;1],f_rosenbrock,g_rosenbrock,1e-12,100,20,20,1e-4,.9,"bt","gd")
#(x, status )= qn(3000,f_square,g_square,1e-12,1000,20,20,1e-4,.9,"wolfe","gd")

x0 = zeros(784,1)

lambda = 0.1
iter = 500

@time (x, status )= qn(x0,f_logreg(X,y,x0,lambda),g_logreg(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"bt","gd")
@time (x, status )= qn(x0,f_logreg(X,y,x0,lambda),g_logreg(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"wolfe","gd")
println(status)
#println(x)
println(status)
#println(x)


println(evaluate(X,y,x))
