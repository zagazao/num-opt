module GD

using LineSearch

export qn

function qn(x0, f, g,eps, maxiter, maxLSiter, maxzoomiter, c1, c2,ls, dir)
    @printf("iter \t\t fval \t\t opt \t\t alp \t\t lsiter \n")
    xk = x0
    oval = Inf
    for i in 1:maxiter
        fval = f(xk)        
        ∇ = g(xk) 
        stoppingCriteria = norm(∇,Inf)
        
        α = 1
        pk = ∇
        iter = 1
        
        if dir == "gd"
            pk = -∇
        elseif dir == "bfgs"
            pk = -∇
        end
        # compute stepsize by linesearch
        lsZiter = 0
        if ls == "bt"
            α,iter = backTrackingLS(f,g,maxLSiter,c1,xk)
        elseif ls == "wolfe"
            α,iter,lsZiter = wolfeLineSearch(f,g,pk,maxLSiter,maxzoomiter,c1,c2,xk) 
        end
        # compute search direction
       
        # verbose
        @printf("%i \t\t %f \t %f \t %f \t\t %i %i \n",i,fval,stoppingCriteria,α,iter,lsZiter)
        # update current point
        xk = xk + α * pk
        # stopping criterion
        if stoppingCriteria < eps
            return (xk,"optimal")
        end
        if abs(oval - fval) < eps
            return (xk, "NO DECREASE")
        end
        oval = fval
    end    
    return (xk,"maxiter")
end

function BFGS()
    H_0 = eye(100)
end

end