module Functions

export f_square,g_square,f_logreg,g_logreg,f_rosenbrock,g_rosenbrock,g_logreg,sub_g_logreg,f_svm,sub_g_svm,sub_g_x_svm,sub_g_x_logreg
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

function f_logreg(X,y,θ,λ)
    return function(θ)
        data = X
        labels = y
        λ = λ
        funcVal = 0
        # M[i:i,1:size(M,2)] -> take i-th column / datapoint
        # size(data,1) = 1000 datapoints
        for i in 1:size(data,1)
            # x = 784*1 column vector
            x_i = data[i:i,1:size(data,2)]'
            funcVal += log(1+ exp(-labels[i]*θ'*x_i))
        end
        funcVal += λ/2*(norm(θ,2)^2)
        return funcVal[1]
    end
end

function g_logreg(X,y,θ,λ)
    return function(θ)
        data = X
        labels = y
        λ = λ
        # i want column vector
        array = zeros(size(data,2)) # 784-column-vector
        for i in 1:size(data,1) # datapoints
        	x_i = data[i:i,1:size(data,2)]' # i-th datapoint
        	exponential = exp(-labels[i]*θ'*x_i)[1]
        	for j in 1:size(data,2) # fill gradient-vector
        	   array[j] = array[j] + (-labels[i]*x_i[j] * exponential)/(1+exponential)
                if i == 1
                    array[j] +=  λ*θ[j]
                end
            end
        end
        return array
    end
end


#TODO: improve by checking sub-gradient-constraint
function sub_g_logreg(X,y,Θ,λ)
    return function(Θ)
      X = X
      y = y
      λ = λ
      # size of data set
      m = size(X,2)
      # pull one data point random
      idx = rand(1:m)
      # compute gradient for this data point
      x_i = X[idx:idx,1:size(X,2)]'
      #@printf("X[%i:%i,1:%i]",idx,idx,size(X,2))
      array = zeros(size(X,2))
      exponential = exp(-y[idx]*Θ'*x_i)[1]
      for j in 1:size(X,2)
         array[j] = array[j] + (-y[idx]*x_i[j] * exponential)/(1+exponential) + λ*Θ[j]
      end
      return array
    end
end

function sub_g_x_logreg()
        return function(x_i,y_i,θ,λ)
              array = zeros(size(x_i,2))
              exponential = exp(-y_i*dot(θ,x_i))
              return (-y_i * x_i * exponential) / ( 1 + exponential ) + λ * θ
        end
end
#=
SVM - Loss-function and gradient
=#

function f_svm(X,y,θ,λ)
    return function(θ)
        X = X
        y = y
        λ = λ
        fval = 0
        # datapoints
        for i in 1:size(X,1)
            x_i = X[i:i,1:size(X,2)]'
            fval += max(0, (1 - y[i]*dot(θ,x_i)))
        end
        fval = fval / size(X,1)
        fval += λ/2 * norm(θ,2)
        return fval
    end
end

function sub_g_svm(X,y,θ,λ)
    return function(Θ)
      X = X
      y = y
      λ = λ
      # size of data set
      m = size(X,2)
      # pull one data point random
      idx = rand(1:m)
      # compute gradient for this data point
      x_i = X[idx:idx,1:size(X,2)]'
      #@printf("X[%i:%i,1:%i]",idx,idx,size(X,2))
      array = zeros(size(X,2))
      val = 1 - (y[idx] * dot(θ,x_i))
      if val < 0
          return λ * θ;
      else
          return -y[idx] * x_i + λ * θ;
      end
    end
end

#=
Compute the gradient for one datapoint for hinge-loss.
=#
function sub_g_x_svm()
    return function(x_i,y_i,θ,λ)
      array = zeros(size(x_i,2))
      val = 1 - (y_i * dot(θ,x_i))
      if val < 0
          return λ * θ;
      else
          return -y_i * x_i + λ * θ;
      end
  end
end

#
# TOOD:

#

#=
Rosenbrock-function and its gradient.
=#
function f_rosenbrock(x_k)
    return 100*(x_k[2]-x_k[1]^2)^2+(1-x_k[1])^2
end


function g_rosenbrock(x_k)
    return [ -400*x_k[1]*(x_k[2]-x_k[1]^2)-2(1-x_k[1]) ; 200*x_k[1]*(x_k[2]-x_k[1]^2) ]
end

end
