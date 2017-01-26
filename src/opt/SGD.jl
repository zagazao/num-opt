
function sgd(X, y, x0, f, g,eps,lambda, maxiter, stepsize,num_data_points)
    @printf("iter \t\t j-pick \t\t fval \t\t\t diff \t\t opt \t\t alp \t\t lsiter \n")

    # safe best try => pk not always descent direction
    f_best = Inf
    x_best = x0
    x_new = x0
    x_k = x0
    oval = Inf
    val_array = zeros(0)
    stop_array = zeros(0)

    for i in 1:maxiter
        # pick idx
        idx = rand(1:num_data_points)

        x_i = X[idx:idx,:]'
        fval = f(x_new)
        gval = g(x_i, y[idx] , x_k, lambda)
        if fval < f_best
            f_best = fval
            x_best = x_new
        end
        stoppingCriteria = norm(gval,Inf)
        append!( val_array, fval )
        append!( stop_array, stoppingCriteria )
    
        x_new = x_k + stepsize * - gval

        x_k = x_new
        @printf("%i \t\t %i \t\t %f \t\t %f \t %e \n",i,idx,fval,(oval-fval),stoppingCriteria)

        oval = fval
    end
    return (x_k,"iter",val_array,stop_array)
end
