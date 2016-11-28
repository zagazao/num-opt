function logLikelihood(X,y,theta,lambda)
	regularizer = lambda/2*norm(theta)^2	
end

function gradLogLikelihood(X,y,theta,lambda)

end

function logregGD(X, y, lambda; epsilon=1e-4, maxiter=1000,maxLSiter=25, verbose=false)
	m = size(X,2) # i have m datapoints
	n = size(X,1) # n = dimension of the problem
	
	# x in r^n*m (input vector) m*n = m zeilen bei n spalten
	# y are the labels (r^m)^t..
	# i have to split X into m vectors/arrays
	#sizeX = size(X)
	#println(sizeX) # sizeX[0] = m ; sizeX[1] =n
	for i in 1:m
    	col = view(X,:,i)
    	println("DATAPOINT :")
		println(string(col) * " <-> " * string(y[i]))
	end
	f(x) = x
	theta =  rand(1:10,n) # r^n
	optimal = false
	for i in 1:maxiter
		# check optimality
		if norm(theta,Inf) < 1 #TODO: use grad(f(theta)) 
			return (theta,"optimal")
		end
		# verbose
		if verbose
			println("Iteration : " + i)
			println("Objective Value : " + i*i)
			println("Stopping criterion value : " + i)
			println("Stepsize of linesearch : " + i)	
			println("Number of linesearch iterations : " + i)		
		end
	end
	return (theta,"maxIter")
end

# create matrix
A = rand(1:10,10,3)
println(A)
# create labels
labels = rand(1:10,10)


#ensure A^m*n => x is length m
println(logregGD(A,labels,""))


logLikelihood("",labels,ones(10),0.1)

