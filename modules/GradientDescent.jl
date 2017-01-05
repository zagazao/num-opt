module GD

include("Linesearch.jl")

using LS

export qn

function qn(x0, f, g,eps, maxiter, maxLSiter, maxzoomiter, c1, c2,ls, dir)
    @printf("iter \t\t fval \t\t\t diff \t\t opt \t\t alp \t\t lsiter \n")
    xk = x0
    x_old = xk
    oval = Inf
    H_k = Inf
    grad_prev = 0
    grad_new = 0
    val_array = zeros(0)
    stop_array = zeros(0)

    grad_new = g(xk)
    for i in 1:maxiter

        fval = f(xk)
        stoppingCriteria = norm(grad_new,Inf)

        append!( val_array, fval )
        append!( stop_array, stoppingCriteria )

        if stoppingCriteria < eps
           return (xk,"optimal",val_array,stop_array)
        end
        if abs(oval - fval) < eps
            #return (xk, "NO DECREASE",val_array,stop_array)
        end

        α = 1
        pk = -grad_new
        iter = 1

        # set search-direction
        if dir == "gd"
            pk = -grad_new
        elseif dir == "bfgs"
            if i == 1
                # n * n identity matrix
                H_0 = eye(size(x0,1))
                H_k = H_0
            else
              # update H_k+1
              s_k = xk - x_old
              # y_k is gradient difference
              y_k = grad_new - grad_prev
              rho = 1 / dot(s_k,y_k)
              Z = eye(size(x0,1)) - rho * y_k * s_k'
              H_k = Z'*H_k*Z  + rho*s_k *s_k'
              #println("H_K")
              #println(H_k)
              #println(size(H_k))
              pk = -H_k * grad_new
              #println(pk)
            end
        end
        # compute stepsize by linesearch
        lsZiter = 0
        if ls == "bt"
            α,iter = backTrackingLS(f,g,maxLSiter,c1,xk,grad_new,fval)
        elseif ls == "wolfe"
            α,iter,lsZiter = wolfeLineSearch(f,g,pk,maxLSiter,maxzoomiter,c1,c2,xk,grad_new,fval)
        end

        @printf("%i \t\t %f \t\t %f \t %e \t %e \t\t %i %i \n",i,fval,(oval-fval),stoppingCriteria,α,iter,lsZiter)
        # update current point
        x_old = xk
        xk = xk + α * pk

        # update values
        grad_prev = grad_new
        grad_new = g(xk)
        oval = fval
        # ∇ = grad_new
    end
    return (xk,"maxiter",val_array,stop_array)
end

end
