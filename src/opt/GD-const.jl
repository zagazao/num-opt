
function gd(X, y, x0, f, g, eps, λ, maxiter, stepsize, num_data)

    x_k = x0
    oval = Inf

    val_array = zeros(0)
    stop_array = zeros(0)

    for i in 1:maxiter

        # Logistic-regression with l2-norm of lambda
        fval = f(x_k)
        gval = g(x_k)

        stoppingCriteria = norm(gval,Inf)
        append!( val_array, fval )
        append!( stop_array, stoppingCriteria )
        @printf("%i \t\t %f \t\t %f \t %e \n",i,fval,(oval-fval),stoppingCriteria)

        x_k = x_k - stepsize * gval
        oval = fval

    end
    return (x_k,"iter",val_array,stop_array)
end
