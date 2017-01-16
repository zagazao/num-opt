include("./src/functions/Functions.jl")
include("./src/evaluate/Evaluate.jl")
include("./src/linesearch/Linesearch.jl")

using MAT

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
        gradient = g_likelihood(theta)
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

dataLocation = "data/mnist67.scale.1k.mat"

# 784 dimensions | 1000 data points
file = matopen(dataLocation)
tmp = read(file, "X")

X = full(tmp)
y = read(file,"y")

theta, state = logregGD(X,y,0,1e-4,1000,20,true)
@printf("Optimization finished with state %s.\n",state)
accuracy = evaluate(X,y,theta)
@printf("Accuracy of solution : %f %%.\n",accuracy*100)

theta, state = logregGD(X,y,0.1,1e-4,1000,20,true)
@printf("Optimization finished with state %s.\n",state)
accuracy = evaluate(X,y,theta)
@printf("Accuracy of solution : %f %%.\n",accuracy*100)

theta, state = logregGD(X,y,100,1e-4,1000,20,true)
@printf("Optimization finished with state %s.\n",state)
accuracy = evaluate(X,y,theta)
@printf("Accuracy of solution : %f %%.\n",accuracy*100)

theta, state = logregGD(X,y,1000,1e-4,1000,20,true)
@printf("Optimization finished with state %s.\n",state)
accuracy = evaluate(X,y,theta)
@printf("Accuracy of solution : %f %%.\n",accuracy*100)
