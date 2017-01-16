
using ProximalOperators

function SAGA(X, y, x0, f, g, eps, λ, maxiter, stepsize, num_data,prox_operator)
    @printf("iter \t\t fval \t\t\t diff \t\t opt \t\t alp \t\t lsiter \n")

    x_k = x0
    x_old = x_k
    oval = Inf
    # Arrays of grad_inf and function_values
    val_array = zeros(0)
    stop_array = zeros(0)

    # M is my matrix of derivatives
    M = zeros(size(x0,1),num_data)
    for i in 1:num_data
        # compute f' for each data_point
        x_i = X[i:i,1:size(X,2)]'
        gval_idx = g( x_i ,y[i] ,x0 ,λ )
        # fill i-th column of M
        for j in 1:size(x0,1)
            M[j:j,i:i] = gval_idx[j]
        end
    end

    @printf("Initial derivatives are initialized.\n")

    for i in 1:maxiter

        fval = f(x_k)

        # Step 1:
        # pick a j uniformly at random. j ∈ [1,n]
        j = rand(1:num_data)
        # pick the j-th datapoint
        x_j = X[j:j,1:size(X,2)]'

        # Step 2:
        # Take phi_j_k_plus = xk and store f'_j(phi_j_k_plus) in the table
        phi_j_k_plus = x_k
        gval_j = g(x_j ,y[j] , phi_j_k_plus, λ)
        # update gval_j in table
        for k in 1:size(x0,1)
            M[k:k,j:j] = gval_j[k]
        end


        # Step 3:
        # Update x using f'_j(phi_j_k_plus),
        # need sum
        sum = zeros(size(x0,1))
        for k in 1:num_data
            sum += M[1:size(x0,1),k:k]
        end
        sum = sum / num_data
        
        # TODO: Try to remove gradient-computation
        w_k_plus = x_k - stepsize * ( M[1:size(x0,1),j:j] - g(x_j, y[j], x_old, 0 ) + sum )

        x_old = x_k
        # Update x_k with prox_operator
        x_k, f_k = prox(prox_operator, w_k_plus, stepsize)
        #x_k = prox{ h(x) + 1 / (2 * stepsize) }
        @printf("%i \t\t %i \t\t %f \t\t %f \n",i,j,fval,(oval-fval))

        #for i in 1:num_data
        #    # compute f' for each data_point
        #    x_i = X[i:i,1:size(X,2)]'
        #    gval_idx = g( x_i ,y[i] ,x0 ,0 )
        #    # fill i-th column of M
        #    for j in 1:size(x0,1)
        #        M[j:j,i:i] = gval_idx[j]
        #    end
        #end
        # No decrease
        #if fval > oval
        #    return (x_k, "iter", maxiter)
        #end
        oval = fval

    end
    return (x_k, "iter", maxiter)
end
