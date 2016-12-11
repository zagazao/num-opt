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

rosenbrock = false
logreg = true

lambda = 0.1
iter = 10

if rosenbrock
    rosen_x0 = zeros(2,1)

    # optimize rosenbrock with backtracking linesearch and default pk
    (x, status,vals, stops )= qn(rosen_x0,f_rosenbrock,g_rosenbrock,1e-12,5000,20,20,1e-4,.9,"bt","gd")

    plotArray(vals,"../plots/rosen-fval.svg")
    plotArray(stops,"../plots/rosen-stop.svg")

    (x, status,vals, stops )= qn(rosen_x0,f_rosenbrock,g_rosenbrock,1e-12,5000,20,20,1e-4,.9,"wolfe","gd")

    plotArray(vals,"../plots/rosen-wolfe-fval.svg")
    plotArray(stops,"../plots/rosen-wolfe-stop.svg")

    (x, status,vals, stops )= qn(rosen_x0,f_rosenbrock,g_rosenbrock,1e-12,5000,20,20,1e-4,.9,"wolfe","bfgs")

    plotArray(vals,"../plots/rosen-bfgs-fval.svg")
    plotArray(stops,"../plots/rosen-bfgs-stop.svg")
end 

if logreg
    x0 = zeros(size(X,2),1)

    if false
        println("Backtracking GradientDescent LogReg")
        @time (x, status,vals,stops ) = qn(x0,f_logreg(X,y,x0,lambda),g_logreg2(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"bt","gd")
        println(evaluate(X,y,x))
        
        plotDoubleArray(vals,stops,"../plots/gd-bt.svg")
        plotArray(vals,"../plots/gd-fval.svg")
        plotArray(stops,"../plots/gd-stop.svg")
    end
    if true
        println("WolfeLS GradientDescent LogReg")
        @time (x, status,vals,stops ) = qn(x0,f_logreg(X,y,x0,lambda),g_logreg2(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"wolfe","gd")
        println(evaluate(X,y,x))

        
        plotDoubleArray(vals,stops,"../plots/gd-bt.svg")
        plotArray(vals,"../plots/wolfe-fval.svg")
        plotArray(stops,"../plots/wolfe-stop.svg")
    end
   
    if false
        println("Wolfe BFGS LogReg")
        @time (x, status,vals,stops ) = qn(x0,f_logreg(X,y,x0,lambda),g_logreg2(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"wolfe","bfgs")
        println(evaluate(X,y,x))

        plotArray(vals,"../plots/bfgs-fval.svg")
        plotArray(stops,"../plots/bfgs-stop.svg")
    end

end
