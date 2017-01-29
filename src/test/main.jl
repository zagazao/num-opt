include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/SAG.jl")
include("../opt/SAGA.jl")
include("../opt/SAGA_PROX.jl")
include("../opt/SGD.jl")
include("../data/data.jl")

using ProximalOperators
using Gadfly
using DataFrames
using ArgParse

#for lambda in [1e-2,1e-3,1e-4,1e-5]
#    for step in [0.1, 0.05, 0.01, 0.005, 0.001]

function main(args)


    lambda = 0
    step = 0
    iter = 300
    mode = "logreg"

    s = ArgParseSettings(description = "Example 1 for argparse.jl: minimal usage.")

    @add_arg_table s begin
        "--step","-s"
        "--lam", "-l"
        "--iter"
        "--mode"
    end
    parsed_args = parse_args(s)
    println("Parsed args:")
    for (key,val) in parsed_args
        if key == "lam"
            lambda = val
        elseif key == "step"
            step = val
        elseif key == "iter"
            iter = val
        elseif key == "mode"
            mode = val
        end
        println("  $key  =>  $(repr(val))")
    end
    println(lambda)
    println(step)
    lambda = parse(Float64,lambda)
    step = parse(Float64,step)
    iter = parse(Int32, iter)
    X, y = getFullData()
    (X_test,y_test) = getTestData()

    # init initial point x_0
    x0 = zeros(size(X,2),1)

    plotname = string("comparison-lam=", lambda,  "-step=",step, "-iter=",iter, "-mode=", mode ,".svg" )
    println(plotname)

    if mode == "logreg"
    f = f_logreg(X,y,x0,lambda)
    g = sub_g_logreg()
    elseif mode == "svm"
        f = f_svm(X,y,x0,lambda)
        g = sub_g_svm()
    else
        error("wrong mode!")
    end

    n = size(X,1)

    prox_operator = NormL2(lambda)

    @time (theta_sag, strings, iter,val_array_sag) = SAG(X,y,x0,f,g,1e-8,lambda,iter,step,n)

    @time (theta_saga, strings, iter,val_array_saga) = SAGA(X,y,x0,f,g,1e-8,lambda,iter,step,n)

    @time (theta_saga_prox, strings, iter,val_array_saga_prox) = SAGA(X,y,x0,f,g,1e-8,lambda,iter,step,n,prox_operator)

    @time (theta_sgd,state,val_array_sgd,stop_array_sgd) = sgd(X,y,x0,f,g,1e-8,lambda,iter,step,n)

    # EVALUATION #

    # evaluate on training set #
    @printf("Accuracy of sag on trainingdata = %f.\n",evaluate(X,y,theta_sag,mode,false))
    @printf("Accuracy of saga on trainingdata = %f.\n",evaluate(X,y,theta_saga,mode,false))
    @printf("Accuracy of saga_prox on trainingdata = %f.\n",evaluate(X,y,theta_saga_prox,mode,false))
    @printf("Accuracy of sgd on trainingdata = %f.\n",evaluate(X,y,theta_sgd,mode,false))

    println()

    # evalute test dat #
    @printf("Accuracy of sag on testdata = %f.\n",evaluate(X_test,y_test,theta_sag,mode,false))
    @printf("Accuracy of saga on testdata = %f.\n",evaluate(X_test,y_test,theta_saga,mode,false))
    @printf("Accuracy of saga_prox on testdata = %f.\n",evaluate(X_test,y_test,theta_saga_prox,mode,false))
    @printf("Accuracy of sgd on testdata = %f.\n",evaluate(X_test,y_test,theta_sgd,mode,false))

    # plot values

    df = DataFrame(iter = [identity(x) for x in 1:iter],
                    obj = val_array_saga,
                    obj2 = val_array_saga_prox,
                    obj3 = val_array_sgd,
                    obj4 = val_array_sag)

    p = plot(layer(df,x=:iter,y=:obj4,Geom.line,Theme(default_color=colorant"green")),
            layer(df,x=:iter,y=:obj,Geom.line,Theme(default_color=colorant"red")),
            layer(df,x=:iter,y=:obj2,Geom.line,Theme(default_color=colorant"blue")),
            layer(df,x=:iter,y=:obj3,Geom.line,Theme(default_color=colorant"orange")))
    img = SVG(plotname, 6inch, 6inch)
    draw(img, p)

end

main(ARGS)
