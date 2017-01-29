#=
Square-function with gradient
=#
function f_square(x)
    return x^2
end

function g_square(x)
    return 2*x
end

#=
Logistic-regression loss-function, gradient and sub-gradient
=#

function f_logreg()
    return function(x_i,y_i,θ,λ)
        return log(1+exp(-y_i*dot(θ,x_i)))
    end
end

function f_logreg(X,y,θ,λ)
    return function(θ)
        data = X
        labels = y
        λ = λ
        funcVal = 0
        # size(data,1) = 1000 datapoints
        for i in 1:size(data,1)
            # x = 784*1 column vector
            x_i = data[i:i,:]'
            funcVal += log( 1+ exp(-labels[i] * dot(θ,x_i) ) )
        end
        funcVal += λ/2*(norm(θ,2)^2)
        return funcVal
    end
end

function g_logreg(X,y,θ,λ)
    return function(θ)
        data = X
        labels = y
        λ = λ
        # n-dim-column-vector
        array = zeros(size(data,2))
        # add l2 norm
        array += λ * θ
        # iterate datapoints
        for i in 1:size(data,1)
            # pick i-th datapoint
        	x_i = data[i:i,:]'
            exponential = exp(-labels[i]*dot(θ,x_i))
            array += (-labels[i]*x_i * exponential)/(1+exponential)
        end
        return array
    end
end

function sub_g_logreg()
        return function(x_i,y_i,θ,λ)
              exponential = exp(-y_i*dot(θ,x_i))
              return (-y_i * x_i * exponential) / ( 1 + exponential ) + λ * θ
        end
end

#=
hinge-loss - Loss-function and gradient
=#
function f_svm(X,y,θ,λ)
    return function(θ)
        X = X
        y = y
        λ = λ
        fval = 0
        # datapoints
        for i in 1:size(X,1)
            x_i = X[i:i,:]'
            fval += max(0, (1 - y[i]*dot(θ,x_i)))
        end
        fval = fval / size(X,1)
        fval += λ/2 * norm(θ,2)
        return fval
    end
end

#=
Compute the gradient for one datapoint for hinge-loss.
=#
function sub_g_svm()
    return function(x_i,y_i,θ,λ)
      val = 1 - (y_i * dot(θ,x_i))
      if val < 0
          return λ * θ;
      else
          return -y_i * x_i + λ * θ;
      end
  end
end

#=
Rosenbrock-function and its gradient.
=#
function f_rosenbrock(x_k)
    return 100*(x_k[2]-x_k[1]^2)^2+(1-x_k[1])^2
end


function g_rosenbrock(x_k)
    return [ -400*x_k[1]*(x_k[2]-x_k[1]^2)-2(1-x_k[1]) ; 200*x_k[1]*(x_k[2]-x_k[1]^2) ]
end
