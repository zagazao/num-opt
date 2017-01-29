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

X, y = getFullData()
(X_test,y_test) = getTestData()

# init initial point x_0
x0 = zeros(size(X,2),1)

iter = 250
mode = "logreg"

long_lambda = [0.0001,0.0001,0.00001]
step_sizes = [0.01,0.005, 0.002, 0.001, 0.0001]

for lambda in [0.0001]
    for step in [0.001,0.0001]
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

          plotname = string("comparison-lam=", lambda,  "-step=",step, "-iter=",iter, "-mode=", mode ,".svg" )
        n = size(X,1)

        l1_prox_operator = NormL1(lambda)
        prox_operator = NormL2(lambda)

        @time (theta_sag, strings, iter,val_array_sag) = SAG(X,y,x0,f,g,1e-8,lambda,iter,step,n)

        @time (theta_saga, strings, iter,val_array_saga) = SAGA(X,y,x0,f,g,1e-8,lambda,iter,step,n)

        @time (theta_saga_prox, strings, iter,val_array_saga_prox) = SAGA(X,y,x0,f,g,1e-8,lambda,iter,step,n,prox_operator)

        @time (theta_saga_prox_l1, strings, iter,val_array_saga_prox_l1) = SAGA(X,y,x0,f,g,1e-8,lambda,iter,step,n,l1_prox_operator)

        @time (theta_sgd,state,val_array_sgd,stop_array) = sgd(X,y,x0,f,g,1e-8,lambda,iter,step,n)

        # EVALUATION #

        # evaluate on training set #
        @printf("Accuracy of sag on trainingdata = %f.\n",evaluate(X,y,theta_sag,mode,false))
        @printf("Accuracy of saga on trainingdata = %f.\n",evaluate(X,y,theta_saga,mode,false))
        @printf("Accuracy of saga_prox on trainingdata = %f.\n",evaluate(X,y,theta_saga_prox,mode,false))
        @printf("Accuracy of saga_prox_1 on trainingdata = %f.\n",evaluate(X,y,theta_saga_prox_l1,mode,false))
        @printf("Accuracy of sgd on trainingdata = %f.\n",evaluate(X,y,theta_sgd,mode,false))

        println()

        # evalute test dat #
        @printf("Accuracy of sag on testdata = %f.\n",evaluate(X_test,y_test,theta_sag,mode,false))
        @printf("Accuracy of saga on testdata = %f.\n",evaluate(X_test,y_test,theta_saga,mode,false))
        @printf("Accuracy of saga_prox on testdata = %f.\n",evaluate(X_test,y_test,theta_saga_prox,mode,false))
        @printf("Accuracy of saga_prox_1 on testdata = %f.\n",evaluate(X_test,y_test,theta_saga_prox_l1,mode,false))
        @printf("Accuracy of sgd on testdata = %f.\n",evaluate(X_test,y_test,theta_sgd,mode,false))

        # plot values

        df = DataFrame(iter = [identity(x) for x in 1:iter],
                        obj = val_array_saga,
                        obj2 = val_array_saga_prox,
                        obj3 = val_array_sgd,
                        obj4 = val_array_sag,
                        obj5 = val_array_saga_prox_l1)

        p = plot(layer(df,x=:iter,y=:obj4,Geom.line,Theme(default_color=colorant"green")),
                layer(df,x=:iter,y=:obj,Geom.line,Theme(default_color=colorant"red")),
                layer(df,x=:iter,y=:obj2,Geom.line,Theme(default_color=colorant"blue")),
                layer(df,x=:iter,y=:obj3,Geom.line,Theme(default_color=colorant"orange")),
                layer(df,x=:iter,y=:obj5,Geom.line, Theme(default_color=colorant"black")))
        img = SVG(plotname, 6inch, 6inch)
        draw(img, p)
    end
end
