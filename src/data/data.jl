using MAT
using MNIST

dataLocation = "data/mnist67.scale.1k.mat"

function update_labels(y)
    for i in 1:size(y,1)
        if y[i] % 2 == 0
             y[i] = 0
        else
             y[i] = 1
        end
    end
end

function getOldData()
    println("Loading dataset.")
    file = matopen(dataLocation)
    tmp = read(file, "X")

    X = full(tmp)
    y = read(file,"y")
    println("Loaded dataset.")
    return (X,y)
end

function getFullData()
    println("Loading dataset.")
    X, y = traindata()
    X = X'
    X = X / 276
    update_labels(y)
    println("Loaded dataset.")
    return (X,y)
end
