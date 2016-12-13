module Evaluate

export evaluate

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
		#@printf("Prediction : %s | Label : %s | Accuracy %f \n",label,labelString,positiveHits / i)
	end
	return positiveHits / size(X,1)
end

end
