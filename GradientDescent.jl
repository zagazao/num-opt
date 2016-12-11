include("modules\\Functions.jl")

using MAT
using Functions

function logregGD(X, y, lambda, epsilon=1e-4, maxiter=1000,maxLSiter=25, verbose=false)
	m = size(X,1) # i have m datapoints
	n = size(X,2) # n = dimension of the problem	
	theta = zeros(784,1)
	multiplier = 0.5
	f_likelihood = f_logreg(X,y,theta,lambda)
    g_likelihood = g_logreg(X,y,theta,lambda)
	optimal = false	
	
	likelihood = Inf
	for i in 1:maxiter
        likelihood = f_likelihood(theta)
		#likelihood = logLikelihood(X,y,theta,lambda)
		#gradient = gradLogLikelihood(X,y,theta,lambda)
        gradient = g_likelihood(theta)
		#gradient = gradient / norm(gradient)
		alpha = 1
		
		# check optimality with inf norm
		if norm(gradient,Inf) < epsilon
			return (theta,"optimal")
		end
		# find step size
		lineSearchIter = 0
        # Linesearch
        for j in 1:maxLSiter        
            f_xk_alpha_k = f_likelihood(theta + ( alpha * -gradient))
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
   
		if verbose		            
            @printf("%i \t\t %f \t %f \t %f \t\t %i \n",i,likelihood,norm(gradient,Inf),alpha,lineSearchIter)
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

theta, state = logregGD(X,y,0.1,1e-5,100,25,true)

println(evaluate(X,y,theta))