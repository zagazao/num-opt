function logLikelihood(X,y,theta,lambda)	
	println(view(X,:,1))
	funcVal = 0	
	for i in 1:size(X,2)
		funcVal += log(1 + exp(-y[i]*theta'*view(X,:,i)))
	end
	funcVal += lambda/2*norm(theta)^2	
end

function gradLogLikelihood(X,y,theta,lambda)
	# compute n gradient values and sum them up.
	array = zeros(0)
	for i in 1:size(X,1)
		tmp = 0
		for j in 1:size(X,2)
			col = view(X,:,j)
			tmp = -y[i]*col[i] * exp(theta'*col)
			tmp = tmp/(1+exp(-y[j]*theta'*view(X,:,j))[1])
		end
		#println(tmp)		
		append!( array, tmp )
	end
	return array
end

function logregGD(X, y, lambda, epsilon=1e-4, maxiter=1000,maxLSiter=25, verbose=false)
	
	# x in r^n*m (input vector) m*n = m zeilen bei n spalten
	m = size(X,2) # i have m datapoints
	n = size(X,1) # n = dimension of the problem	
	#theta =  rand(1:10,n) # r^n
	theta = rand(n)
	alpha = 0.1
	multiplier = 0.5
	
	for i in 1:m
    	col = view(X,:,i)
    	println("DATAPOINT :")
		println(string(col) * " <-> " * string(y[i]))
	end
	f(x) = x
	optimal = false
	for i in 1:maxiter
		# check optimality
		#if norm(theta,Inf) < 1 #TODO: use grad(f(theta)) 
	#		return (theta,"optimal")
	#	end
		
		# verbose
		if verbose
		
			likelihood = logLikelihood(X,y,theta,lambda)
			println("Iteration : " * string(i))
			println("Objective Value : " * string(likelihood))
			println("Stopping criterion value : " * string(norm(theta,Inf)))
			println("Stepsize of linesearch : " * string(alpha))	
			println("Number of linesearch iterations : " * string(i))
			println("Thetha" * string(theta))
		end
		# descent
		#TODO: schrink alpha
		println("VAL")		
		val = gradLogLikelihood(X,y,theta,lambda)
		println(val)
		println("ALPHA*VAL")
		println(alpha*val)
		println("THETA ")
		println(theta)
		println("THETA + VAL")
		println(theta + val)
		theta = theta + val
		#theta +=
		#theta += val * alpha
		# x_k+1 = x_k + alpha_K*p_k
		# p_k search direction
		# alpha_k stepsize
		println("THETA ")
		println(theta)
	end
	return (theta,"maxIter")
end

numDataPoints = 100
numDimension = 20

# create matrix
A = rand(1:10,numDimension,numDataPoints) # create dataset
println(A)
# create labels
labels = rand(0:1,1,numDataPoints)

println("LABELS")
println(labels)
# n dimension column vector
randomTheta = rand(numDimension,1)
println("INIT THETA")
println(randomTheta)


#ensure A^m*n => x is length m
println(logregGD(A,labels,0.1,1e-4,5,25,true))
#println(logregGD(A,labels,0.1,1e-4,1000,25,true))
#println(logLikelihood(A,labels,randomTheta,0.1,1e-4,1000,25,true))

#println(logLikelihood(A,labels,randomTheta,0.1,1e-4,10,25,true))

