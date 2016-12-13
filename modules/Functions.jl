module Functions

export f_square,g_square,f_logreg,g_logreg,f_rosenbrock,g_rosenbrock,g_logreg2

function f_square(x)
    return x^2
end

function g_square(x)
    return 2*x
end

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
        array = zeros(0)
        # steepest direction for each dimension r^n = size(data,2)
        for i in 1:size(data,2)
            tmp = 0
            # col = 784*1 column vector
            x_i = data[i:i,1:size(data,2)]'
            exponential = exp(-labels[i]*θ'*x_i)
            tmp = (-labels[i]*x_i[i] * exponential)/(1+exponential) + λ*θ[i]
            append!( array, tmp )
        end
        return array
    end
end

function g_logreg2(X,y,θ,λ)
    return function(θ)
        data = X
        labels = y
        λ = λ
        # i want column vector
        array = zeros(size(data,2))
        for i in 1:size(data,1)
        	x_i = data[i:i,1:size(data,2)]'
        	exponential = exp(-labels[i]*θ'*x_i)[1]
        	for j in 1:size(data,2)
        	   array[j] = array[j] + (-labels[i]*x_i[j] * exponential)/(1+exponential)
                if i == 1
                    array[j] +=  λ*θ[j]
                end
            end

        end
        return array
    end
end

function f_rosenbrock(x_k)
    return 100*(x_k[2]-x_k[1]^2)^2+(1-x_k[1])^2
end


function g_rosenbrock(x_k)
    return [ -400*x_k[1]*(x_k[2]-x_k[1]^2)-2(1-x_k[1]) ; 200*x_k[1]*(x_k[2]-x_k[1]^2) ]
end

end
