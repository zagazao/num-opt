module Functions

export f_square,g_square,f_logreg,g_logreg,f_rosenbrock,g_rosenbrock

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
            # x = 784*1 vector => column vector
            x = data[i:i,1:size(data,2)]'
            funcVal += log(1+ exp(-labels[i]*θ'*x)) + λ/2*norm(θ,2)^2
        end
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
            col = data[i:i,1:size(data,2)]'
            exponential = exp(-labels[i]*θ'*col)
            tmp = (-labels[i]*col[i] * exponential)/(1+exponential) + λ*θ[i]
            append!( array, tmp )
        end
        return array
    end
end

function f_rosenbrock(x_k)
    return 100*(x_k[2]-x_k[1]^2)^2+(1-x_k[1])^2
end

function g_rosenbrock(x_k)
    return [2(200x_k[1]^3-200x_k[1]*x_k[2]+1-x_k[1]); 200*x_k[2]-200*x_k[1]^2]
end

end