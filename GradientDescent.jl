using MAT

function getFunction(X,y,theta,lambda)
    return function(theta)
        data = X
        labels = y
		lambda = lambda
        funcVal = 0	
        # compute likelihood for each data-point
        for i in 1:size(data,1)
            x = data[i:i,1:size(data,2)]
            funcVal += log(1+ exp(-labels[i]*theta'*x')) + lambda/2*norm(theta,2)^2
        end
        return funcVal[1]
    end
end

function logLikelihood(X,y,theta,lambda)
	funcVal = 0	
	# compute likelihood for each data-point
	for i in 1:size(X,1)
		x = X[i:i,1:size(X,2)]
		funcVal += log(1+ exp(-y[i]*theta'*x')) + lambda/2*norm(theta,2)^2
	end
	return funcVal[1]
end

function gradLogLikelihood(X,y,theta,lambda)
	# compute n gradient values and sum them up.
	array = zeros(0)
	for i in 1:size(X,2)
		tmp = 0
		col = X[i:i,1:size(X,2)]
		exponential = exp(-y[i]*theta'*col')
		tmp = (-y[i]*col[i] * exponential)/(1+exponential)
		append!( array, tmp )
	end
	return array
end

function getGradFunc(X,y,theta,lambda)
	return function(theta)
		data = X
		labels = y
		lambda = lambda
		array = zeros(0)
		for i in 1:size(data,2)
			tmp = 0
			col = data[i:i,1:size(data,2)]
			exponential = exp(-labels[i]*theta'*col')
			tmp = (-labels[i]*col[i] * exponential)/(1+exponential) + lambda*theta[i]
			append!( array, tmp )
		end
		return array
	end
end

function logregGD(X, y, lambda, epsilon=1e-4, maxiter=1000,maxLSiter=25, verbose=false)
	# x in r^n*m (input vector) m*n = m zeilen bei n spalten
	m = size(X,1) # i have m datapoints
	n = size(X,2) # n = dimension of the problem	
	theta = zeros(784,1)
	multiplier = 0.5
	
	optimal = false	
	
	likelihood = Inf
	for i in 1:maxiter
	
		likelihood = logLikelihood(X,y,theta,lambda)
		gradient = gradLogLikelihood(X,y,theta,lambda)
		#gradient = gradient / norm(gradient)
		alpha = 1
		
		# check optimality with inf norm
		if norm(gradient,Inf) < epsilon
			return (theta,"optimal")
		end
		# find step size
		lineSearchIter = 0
        #@printf("----------------------------------------\n")
		for j in 1:maxLSiter
		
			#@printf("Grad-norm : %f\n",norm(gradient,1))
			f_xk_alpha_k = logLikelihood(X,y,theta+(alpha*-gradient),lambda)
			f_xk_plus_grad = likelihood + (epsilon*alpha*gradient' * -gradient)[1]
			#@printf("%f \t < \t %f\n",f_xk_alpha_k,f_xk_plus_grad)
		
			if  ! (f_xk_alpha_k < f_xk_plus_grad)
				# stop if left smaller than right
				alpha = alpha * multiplier		
				lineSearchIter += 1
			else 
				break
			end	
		end
		# verbose
		println(typeof(likelihood))
		if verbose		
            
            @printf("%i \t\t %f \t %f \t %f \t\t %i \n",i,likelihood,norm(gradient,Inf),alpha,lineSearchIter)
			#println("Iteration : " * string(i))
			#println("Objective Value : " * string(likelihood))
			#println("Stopping criterion value : " * string(norm(gradient,Inf)))
			#println("Stepsize of linesearch : " * string(alpha))	
			#println("Number of linesearch iterations : " * string(lineSearchIter))
		end		
		# descent
		theta = theta + alpha * -gradient
	end
	return (theta,"maxIter")
end

function predict(theta,x)
	return 1 / (1+exp(-theta'*x')[1])
end

function evaluate(X,y,theta)
	positiveHits = 0
	for i in 1:size(X,1)
		prediction = predict(theta,X[i:i,1:size(X,2)])
		label = 1
		if prediction < 0.5 
			label = -1
		end
		
		if label == y[i]
			positiveHits += 1
		end
		if label == 1 
			label = "+1"
		end
		labelString = "-1.0"
		if y[i] == 1.0
			labelString = "+1.0"
		end
		@printf("Prediction : %s | Label : %s | Accuracy %f \n",label,labelString,positiveHits / size(X,1))
	end
	return positiveHits / size(X,1)
end
dataLocation = "data/mnist67.scale.1k.mat"

# 784 dimensions
# 1000 data points
file = matopen(dataLocation)
tmp = read(file, "X")

X = full(tmp)
y = read(file,"y")

theta, state = logregGD(X,y,0.1,1e-4,100,25,true)

println(evaluate(X,y,theta))