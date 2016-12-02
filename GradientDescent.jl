function logLikelihood(X,y,theta,lambda)	
	println(view(X,:,1))
	funcVal = 0	
	for i in 1:size(X,1)
		funcVal += log(1 + exp(-y[i]*theta*X[i:i,1:size(X,2)]))
	end
	funcVal += lambda/2*norm(theta)^2	
end

function gradLogLikelihood(X,y,theta,lambda)
	# compute n gradient values and sum them up.
	array = zeros(0)
	for i in 1:size(X,1)
		tmp = 0
		for j in 1:size(X,2)
			col = X[i:i,1:size(X,2)]
			tmp = -y[i]*col[i] * exp(theta*col)
			#tmp = tmp/(1+exp(-y[j]*theta'*view(X,:,j))[1])
		end
		#println(tmp)		
		append!( array, tmp )
	end
	return array
end

function logregGD(X, y, lambda, epsilon=1e-4, maxiter=1000,maxLSiter=25, verbose=false)
	
	# x in r^n*m (input vector) m*n = m zeilen bei n spalten
	m = size(X,1) # i have m datapoints
	n = size(X,2) # n = dimension of the problem	
	#theta =  rand(1:10,n) # r^n
	theta = rand(n)
	alpha = 0.1
	multiplier = 0.5
	
	for i in 1:n
    #	col = view(X,:,i)
    #	println("DATAPOINT :")
	#	println(string(col) * " <-> " * string(y[i]))
	end
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

dataLocation = "data/mnist67.scale.1k.mat"

using MAT

# 784 dimensions
# 1000 data points
file = matopen(dataLocation)
tmp = read(file, "X")

X = full(tmp)
y = read(file,"y")

println(logregGD(X,y,0.1,1e-4,5,25,true))

println(length(y))

