
function sgd(x0, f, g,eps, maxiter, stepsize)
    @printf("iter \t\t fval \t\t\t diff \t\t opt \t\t alp \t\t lsiter \n")

    xk = x0
    oval = Inf
    H_k = Inf
    grad_new = 0
    val_array = zeros(0)
    stop_array = zeros(0)

    ∇ = g(xk)
    for i in 1:maxiter
      fval = f(xk)
      stoppingCriteria = norm(∇,Inf)
      if fval > oval
        stepsize = stepsize * 0.5
      end
      append!( val_array, fval )
      append!( stop_array, stoppingCriteria )

      if stoppingCriteria < eps
         return (xk,"optimal",val_array,stop_array)
      end
      if abs(oval - fval) < eps
          #return (xk, "NO DECREASE",val_array,stop_array)
      end
      xk = xk + stepsize * - ∇
      @printf("%i \t\t %f \t\t %f \t %e \n",i,fval,(oval-fval),stoppingCriteria)
      oval = fval
    end

end
