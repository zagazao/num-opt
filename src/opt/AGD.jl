
using ProximalOperators

function agd(X, y, x0, f, g, eps, λ, maxiter, stepsize, num_data,prox_operator)

    x_k = x0
    x_old = x0
    oval = Inf

    val_array = zeros(0)
    stop_array = zeros(0)

    gval = g(x_k)
    for i in 1:maxiter

        # Logistic-regression with l2-norm of lambda
        fval = f(x_k)

        y_k = x_k + (i-1) / (i+2) * (x_k-x_old)
        gval = g(y_k)
        z_k = y_k - stepsize * gval

        x_old = x_k
        x_k, f_k = prox(prox_operator, z_k, stepsize)
        oval = fval


        stoppingCriteria = norm(gval,Inf)
        append!( val_array, fval )
        append!( stop_array, stoppingCriteria )
        @printf("%i \t\t %f \t\t %f \t %e \n",i,fval,(oval-fval),stoppingCriteria)

    end
    return (x_k,"iter",val_array,stop_array)
end
