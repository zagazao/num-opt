module Plotting

export plotLogReg,plotArray

using Gadfly
using RDatasets
using DataFrames

iris = dataset("datasets", "iris")

function plotArray(array,filename)
    df = DataFrame(iter = [identity(x) for x in 1:size(array,1)], obj = array)
    p = plot(df,x=:iter,y=:obj,Geom.line)
    img = SVG(filename, 6inch, 4inch)
    draw(img, p)
end

function plotLogReg(X) 
    p = plot(iris, x=:X, y=:y, Geom.line)
    img = SVG("iris_plot.svg", 6inch, 4inch)
    draw(img, p)
end
#END MODULE
end