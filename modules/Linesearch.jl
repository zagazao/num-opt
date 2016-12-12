module LineSearch

export backTrackingLS,wolfeLineSearch

function backTrackingLS(f,g,maxLSiter,c1,θ,∇,fval)
    # θ is column vector
    α = 1
    lineSearchIter = 0 
    for i in 1:maxLSiter
        f_xk_alpha_k = f( θ + α * -∇)
        f_xk_plus_grad = fval + ( c1 * α * ∇' * -∇)[1] 
		#@printf("%f \t < \t %f\n",f_xk_alpha_k,f_xk_plus_grad)       
        if  ! (f_xk_alpha_k < f_xk_plus_grad[1])
            α = α * 0.5
            lineSearchIter += 1
        else 
            break
        end
    end
    return (α,lineSearchIter)
end

# function / gradient-function / direction / ls iter / zoom iter / c1 / v2 / theta
function wolfeLineSearch(f,g,pk,maxLSiter,maxZoomIter,c1,c2,xk,grad,fval) 
    # NEED ITERATION COUNTER
    a_0 = 0.0
    a_iminus1 = a_0    
    a_i = 1
    a_max = 65536.0
    
    phi_0 = fval    
    phi_a_iminus1 = phi_0
    phi_a_i = NaN

    phi_prime_0 = (pk'*grad)[1] 
    phi_prime_a_i = NaN
    
    for i in 1:maxLSiter
        #@printf("zoom : ( ai-1 = %f , a1 = %f )\n",a_iminus1,a_i)
        
        phi_a_i = f(xk + a_i*pk)
        while isinf(phi_a_i)
            println("isinf")
            a_i = a_i * 0.5
            phi_a_i = f(xk + a_i*pk)
        end
        # check wolfe condition
        if phi_a_i > phi_0 + c1 * a_i * phi_prime_0 ||
            (i > 1 && phi_a_i >= phi_a_iminus1)
            (a_star,iter) = zoom(a_iminus1,a_i,maxZoomIter,phi_0,phi_prime_0,f,g,pk,xk,c1,c2)
            return (a_star, i , iter)
        end
        
        phi_prime_a_i = (pk'*g(xk + a_i*pk))[1] 
        # check condition
        if abs(phi_prime_a_i) <= -c2*phi_prime_0
            #println("BREAK")
            return (a_i,i,0)
        end
        
        if phi_prime_a_i >= 0
            (a_star,iter) = zoom(a_i,a_iminus1,maxZoomIter,phi_0,phi_prime_0,f,g,pk,xk,c1,c2)  
            return (a_star, i , iter)         
        end
        
        # choose alphai+1 in (a_i,a_max)
        a_iminus1 = a_i
        a_i = a_i * 2
                
        phi_a_iminus1 = phi_a_i
    end
    return (a_i,maxLSiter,0)
end

function zoom(alpha_l, alpha_u,maxzoomiter,phi_0,phi_prime_0,f,g,pk,xk,c1,c2)
   
   phi_prime_al = (pk'*g(xk + alpha_l * pk))[1]
   phi_prime_au = (pk'*g(xk + alpha_u * pk))[1]
   
   @printf("phi_prime_al = %f, phi_prime_a_u = %f\n",phi_prime_al,phi_prime_au) 
   @printf(" (%f , %f )\n",alpha_l, alpha_u)
   
   for i in 1:maxzoomiter       
        phi_al = f(xk + alpha_l*pk)
        phi_au = f(xk + alpha_u*pk)
        
        #@printf("phi_al = %f, phi_au = %f\n",phi_al,phi_au)
        
        d1 = phi_prime_al + phi_prime_au - 3 * (phi_al - phi_au) / (alpha_l - alpha_u)
        d2 = sqrt(d1*d1-phi_prime_al*phi_prime_au) # WTF is sign(a+b))
       
        #@printf("%f \n",(phi_prime_au - phi_prime_al + 2*d2))
                
        alpha_j = alpha_u - (alpha_u-alpha_l) * (phi_prime_au + d2 - d1) / (phi_prime_au - phi_prime_al + 2*d2)
       
        # check for NaN because too big alpha_u
               
        @printf("| iter = %i | alpha_j = %f | (%f , %f) |d1 = %f | d2 = %f |\n",i,alpha_j,alpha_l,alpha_u,d1,d2)   
        
        phi_alpha_j = f( xk + alpha_j * pk)
        # check condition 
        if phi_alpha_j > phi_0 + c1 * alpha_j * phi_prime_0 || phi_alpha_j >= phi_al
            alpha_u = alpha_j
        else
            phi_prime_aj = (pk' * g(xk + alpha_j * pk))[1]
            if abs(phi_prime_aj) <= -c2*phi_prime_0
                return (alpha_j,i) 
            end
            if phi_prime_aj* (alpha_u-alpha_l) >= 0
                alpha_u = alpha_l
            end
            alpha_l = alpha_j 
        end
        
        if i == maxzoomiter  
            return (alpha_j,maxzoomiter)
        end      
        phi_prime_al = (pk'*g(xk + alpha_l))[1]        
        phi_prime_au = (pk'*g(xk + alpha_u))[1]
        
    end
end

# MODULE END
end