
using ProximalOperators

function SAGA(X, y, x0, f, g, eps, λ, maxiter, stepsize, num_data,prox_operator)

    x_k = x0
    x_old = x_k
    oval = Inf
    fval = Inf
    # Arrays of grad_inf and function_values
    val_array = zeros(0)

    # M is my matrix of derivatives (column i = f_i'(x))
    M = zeros(size(x0,1),num_data)
    @time for i in 1:num_data
        # compute f' for each data_point at x0
        x_i = X[i:i,:]'
        gval_idx = g( x_i ,y[i] ,x0 ,0 )
        # fill i-th column of M
        M[:,i:i] = gval_idx
    end

    @printf("Initial derivatives are initialized.\n")
    @printf("Stepsize was choosen to α = %f.\n", stepsize )
    @printf("iter \t\t j-pick \t\t fval \t\t\t diff \t\t opt\n")

    @time for i in 1:maxiter

        # Step 1:
        # pick a j uniformly at random. j ∈ [1,n]
        j = rand(1:num_data)
        # pick the j-th datapoint
        x_j = X[j:j,:]'

        # Step 2:
        # Take phi_j_k_plus = xk and store f'_j(phi_j_k_plus) in the table
        phi_j_k_plus = x_k
        gval_j = g(x_j ,y[j] , phi_j_k_plus,0)
        stoppingCriteria = norm(gval_j,Inf)
        # update gval_j in table
        old_gval_j = M[:,j]
        M[:,j:j] = gval_j

        # Step 3:
        # Update x using f'_j(phi_j_k_plus)
        sum = zeros(size(x0,1))
        for k in 1:num_data
            sum += M[:,k:k]
        end
        sum = sum / num_data
        w_k = x_k - stepsize * ( gval_j - old_gval_j + sum )
        x_old = x_k
        # Update x_k with prox_operator
        x_k, f_k = prox(prox_operator, w_k, stepsize)

        oval = fval
        fval = f(x_k) # +lambda*norm(x_k,1)
        append!( val_array, fval )
        @printf("%i \t\t %i \t\t %f \t\t %f \t\t %e \n",i,j,fval,(oval-fval),stoppingCriteria)
    end
    return (x_k, "iter", maxiter,val_array)
end
