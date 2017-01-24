include("../functions/Functions.jl")
include("../evaluate/Evaluate.jl")
include("../opt/SAG.jl")
include("../opt/SAGA.jl")
include("../opt/SAGA_PROX.jl")
include("../opt/sgd.jl")
include("../data/data.jl")
include("../plot/Plotting.jl")

using ProximalOperators
using Plotting
using Gadfly
using DataFrames

X, y = getFullData()
(X_test,y_test) = getTestData()

# init initial point x_0
x0 = zeros(size(X,2),1)

#lambda = 0.0001

iter = 300
#step =0.008
sag_step=0.01

mode = "logreg"

for lambda in [1e-2,1e-3,1e-4,1e-5]
    for step in [0.1, 0.05, 0.01, 0.005, 0.001]
        plotname = string("sag - saga - prox - sgd - ", lambda,  " - ",step, ".svg" )
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

        prox_operator = NormL1(lambda)

        @time (theta_sag, strings, iter,val_array_sag) = SAG(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1))

        @time (theta_saga, strings, iter,val_array_saga) = SAGA(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1))

        @time (theta_saga_prox, strings, iter,val_array_saga_prox) = SAGA(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1),prox_operator)

        @time (theta_sgd,state,val_array,stop_array) = sgd(X,y,x0,f,g,1e-8,lambda,iter,step,size(X,1))

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

        #plotTripleArray(val_array_saga,val_array_saga_prox,val_array, "saga-sgd.svg")

        # plot values

        df = DataFrame(iter = [identity(x) for x in 1:iter],
                        obj = val_array_saga,
                        obj2 = val_array_saga_prox,
                        obj3 = val_array,
                        obj4 = val_array_sag)


        p = plot(layer(df,x=:iter,y=:obj4,Geom.line,Theme(default_color=colorant"green")),
                layer(df,x=:iter,y=:obj,Geom.line,Theme(default_color=colorant"red")),
                layer(df,x=:iter,y=:obj2,Geom.line,Theme(default_color=colorant"blue")),
                layer(df,x=:iter,y=:obj3,Geom.line,Theme(default_color=colorant"orange")))
        img = SVG(plotname, 6inch, 6inch)
        draw(img, p)

    end
end
