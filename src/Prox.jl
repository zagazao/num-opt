# This file contains some implementations of prox-operators

# L1 - PROX - OPERATOR
# Param : vector to
function L1prox(x, lambda)
    prox = zeros(size(x,1))
    for index in 1:size(x,1)
        if x[index] < -lambda
            prox[index] = x[index] + lamda
        elseif x[index] > lambda
            prox[index] = x[index] - lambda
        else
            prox[index] = 0
        end
    end
    return prox
end


# L2 - PROX - OPERATOR
# Param :
#   - x = vector to apply proxity
#   - λ = regularization parameter
function L2prox(x,lambda)
    # create empty prox vector
    prox = zeros(size(x,1))
    l2norm = norm(x,2)
    if l2norm > lambda
        return (1-lambda/l2norm)*x 
    else
        return zeros(size(x,1))
    end
end
