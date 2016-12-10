module Plotting

export plotLogReg

using Gadfly
using RDatasets

iris = dataset("datasets", "iris")

function plotLogReg(X) 
    p = plot(iris, x=:X, y=:y, Geom.point)
    img = SVG("iris_plot.svg", 6inch, 4inch)
    draw(img, p)
end
#END MODULE
end