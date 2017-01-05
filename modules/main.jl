include("./Linesearch.jl")
include("./Functions.jl")
include("./GradientDescent.jl")
include("./Evaluate.jl")
include("./SGD.jl")
include("./Plotting.jl")

using Functions
using MAT
using GD
using Plotting

dataLocation = "../data/mnist67.scale.1k.mat"

file = matopen(dataLocation)
tmp = read(file, "X")

X = full(tmp)
y = read(file,"y")
println("Loaded dataset")

rosenbrock = false
logreg = false
sgd_opt = false
svm = true

if svm
    x0 = ones(size(X,2),1)
    f = f_svm(X,y,x0,1)
    g = sub_g_svm(X,y,x0,1)
    sgd_iter = 3000;
    sgd_step = 0.001 #1 / sgd_iter;
    (theta, state, vals, stops ) = @time sgd(x0,f,g,1e-8,sgd_iter,sgd_step)
    println(evaluate(X,y,theta,"svm",true))
end

if sgd_opt
    sgd_iter = 10000;
    sgd_step = 0.001;
    x0 = zeros(size(X,2),1)
    @time sgd(x0,f_logreg(X,y,x0,10),sub_g_logreg(X,y,x0,10),1e-8,sgd_iter,sgd_step)
end

lambda = 0.1
iter = 100

if rosenbrock
    rosen_x0 = zeros(2,1)

    # optimize rosenbrock with backtracking linesearch and default pk
    (x, status,vals, stops )= qn(rosen_x0,f_rosenbrock,g_rosenbrock,1e-12,iter,20,20,1e-4,.9,"bt","gd")

    plotArray(vals,"../plots/rosen-fval.svg")
    plotArray(stops,"../plots/rosen-stop.svg")

    (x, status,vals, stops )= qn(rosen_x0,f_rosenbrock,g_rosenbrock,1e-12,iter,20,20,1e-4,.9,"wolfe","gd")

    plotArray(vals,"../plots/rosen-wolfe-fval.svg")
    plotArray(stops,"../plots/rosen-wolfe-stop.svg")

    (x, status,vals, stops )= qn(rosen_x0,f_rosenbrock,g_rosenbrock,1e-12,iter,20,20,1e-4,.9,"wolfe","bfgs")

    plotArray(vals,"../plots/rosen-bfgs-fval.svg")
    plotArray(stops,"../plots/rosen-bfgs-stop.svg")
end


if logreg
    x0 = zeros(size(X,2),1)

    if true
        println("Backtracking GradientDescent LogReg")
        @time (x, status,vals1,stops1 ) = qn(x0,f_logreg(X,y,x0,lambda),g_logreg(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"bt","gd")
        println(evaluate(X,y,x))

        #plotDoubleArray(vals,stops,"../plots/gd-bt.svg")
        #plotArray(vals,"../plots/gd-fval.svg")
        #plotArray(stops,"../plots/gd-stop.svg")
    end
    if true
        println("WolfeLS GradientDescent LogReg")
        @time (x, status,vals2,stops2 ) = qn(x0,f_logreg(X,y,x0,lambda),g_logreg(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"wolfe","gd")
        println(evaluate(X,y,x))

        #plotDoubleArray(vals,stops,"../plots/gd-bt.svg")
        #plotArray(vals,"../plots/wolfe-fval.svg")
        #plotArray(stops,"../plots/wolfe-stop.svg")
    end

    if true
        println("Wolfe BFGS LogReg")
        @time (x, status,vals3,stops3 ) = qn(x0,f_logreg(X,y,x0,lambda),g_logreg(X,y,x0,lambda),1e-12,iter,20,20,1e-4,.9,"wolfe","bfgs")
        println(evaluate(X,y,x))

        #plotDoubleArray(vals,stops,"../plots/wolfe-bfgs.svg")

        #plotArray(vals,"../plots/bfgs-fval.svg")
        #plotArray(stops,"../plots/bfgs-stop.svg")
    end
    plotTripleArray(vals1,vals2,vals3,vals-lg.svg)
    plotTripleArray(stops1,stops2,stops3,stops-lg.svg)

end
