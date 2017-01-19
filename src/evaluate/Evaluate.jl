function predict_svm(θ,x)
	return dot(θ,x) > 0 ? 1 : -1
end

function predict_logreg(theta,x)
	prediction =  1 / (1+exp(dot(theta,x)))
	if prediction  < 0.5
		return -1
	else
		return 1
	end
end

function evaluate(X,y,theta,mode,verbose=false)
	positiveHits = 0
	for i in 1:size(X,1)
		if mode == "logreg"
			label = predict_logreg(theta,X[i:i,:])
		elseif mode == "svm"
			label = predict_svm(theta,X[i:i,:])
		else
			error("Prediction mode not aviable")
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
		if verbose
			@printf("Prediction : %s | Label : %s | Accuracy %f \n",label,labelString,positiveHits / i)
		end
	end
	return positiveHits / size(X,1)
end
