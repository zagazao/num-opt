module GD

include("Linesearch.jl")

using LS

export qn

function qn(x0, f, g,eps, maxiter, maxLSiter, maxzoomiter, c1, c2,ls, dir)
    @printf("iter \t\t fval \t\t\t diff \t\t opt \t\t alp \t\t lsiter \n")
    xk = x0
    oval = Inf
    if dir == "bfgs"
        # n * n identity matrix
        H_0 = eye(size(x0,1))
        H_k = H_0
    end
    grad_new = 0
    val_array = zeros(0)
    stop_array = zeros(0)

    ∇ = g(xk)
    for i in 1:maxiter

        fval = f(xk)
        stoppingCriteria = norm(∇,Inf)
        append!( val_array, fval )
        append!( stop_array, stoppingCriteria )

        if stoppingCriteria < eps
           return (xk,"optimal",val_array,stop_array)
        end
        if abs(oval - fval) < eps
            return (xk, "NO DECREASE",val_array,stop_array)
        end

        α = 1
        pk = -∇
        iter = 1

        # set search-direction
        if dir == "gd"
            pk = -∇
        elseif dir == "bfgs"
            pk = -H_k * ∇
        end
        # compute stepsize by linesearch
        lsZiter = 0
        if ls == "bt"
            α,iter = backTrackingLS(f,g,maxLSiter,c1,xk,∇,fval)
        elseif ls == "wolfe"
            α,iter,lsZiter = wolfeLineSearch(f,g,pk,maxLSiter,maxzoomiter,c1,c2,xk,∇,fval)
        end

        @printf("%i \t\t %f \t\t %f \t %e \t %e \t\t %i %i \n",i,fval,(oval-fval),stoppingCriteria,α,iter,lsZiter)
        # update current point
        xk = xk + α * pk

        # update values
        grad_new = g(xk)
        if dir == "bfgs"
            # update H_k+1
            # sk = x_k+1 - xk == xk + alpha * - grad - xk = alpha * - grad
            s_k = α * pk
            y_k = grad_new - ∇
            H_k = H_k - (H_k * s_k * s_k' * H_k) / (s_k' * H_k * s_k) + (y_k * y_k') / ( y_k' * s_k)
            if H_k == H_0
                println("Still identitymatrix...")
            end
        end
        oval = fval
        ∇ = grad_new
    end
    return (xk,"maxiter",val_array,stop_array)
end

end
