
function sgd(x0, f, g,eps, maxiter, stepsize,num_data_points,lambda)
    @printf("iter \t\t fval \t\t\t diff \t\t opt \t\t alp \t\t lsiter \n")

    # safe best try => pk not always descent direction
    f_best = Inf
    x_best = x0
    x_new = x0
    xk = x0
    oval = Inf
    val_array = zeros(0)
    stop_array = zeros(0)

    for i in 1:maxiter
        # pick idx
        idx = rand(1:num_data_points)

        x_i = X[idx:idx,1:size(X,2)]'
        fval = f(x_new)
        gval = g(x_i, y[idx] , xk, lambda)
        if fval < f_best
            f_best = fval
            x_best = x_new
            xk = x_new
        end
        stoppingCriteria = norm(gval,Inf)
        append!( val_array, fval )
        append!( stop_array, stoppingCriteria )
        if stoppingCriteria < eps
            #return (xk,"optimal",val_array,stop_array)
        end
        if abs(oval - fval) < eps
            #return (xk, "NO DECREASE",val_array,stop_array)
        end
        x_new = xk + stepsize * - gval
        @printf("%i \t\t %f \t\t %f \t %e \n",i,fval,(oval-fval),stoppingCriteria)
        oval = fval
    end
    return (xk,"iter",val_array,stop_array)
end
