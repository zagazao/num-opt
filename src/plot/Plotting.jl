module Plotting

export plotLogReg,plotArray,plotDoubleArray,plotTripleArray,plotQuadArray

using Gadfly
using DataFrames
using RDatasets

iris = dataset("datasets", "iris")

function plotDoubleArray(array1,array2,filename)
    df = DataFrame(iter = [identity(x) for x in 1:size(array1,1)], obj = array1,obj2 = array2)
    p = plot(layer(df,x=:iter,y=:obj,Geom.line,Theme(default_color=colorant"red")),
            layer(df,x=:iter,y=:obj2,Geom.line,Theme(default_color=colorant"blue")))
    img = SVG(filename, 10inch, 10inch)
    draw(img, p)
end

function plotTripleArray(array1,array2,array3,filename)
    df = DataFrame(iter = [identity(x) for x in 1:size(array1,1)], obj = array1,obj2 = array2,obj3 = array3)
    p = plot(layer(df,x=:iter,y=:obj,Geom.line,Theme(default_color=colorant"red")),
            layer(df,x=:iter,y=:obj2,Geom.line,Theme(default_color=colorant"blue")),
            layer(df,x=:iter,y=:obj3,Geom.line,Theme(default_color=colorant"orange")))
    img = SVG(filename, 10inch, 10inch)
    draw(img, p)
end

function plotQuadArray(array1,array2,array3,array4,filename)
    df = DataFrame(iter = [identity(x) for x in 1:size(array1,1)], obj = array1,obj2 = array2,obj3 = array3,obj4 = array4)
    p = plot(layer(df,x=:iter,y=:obj,Geom.line,Theme(default_color=colorant"red")),
            layer(df,x=:iter,y=:obj2,Geom.line,Theme(default_color=colorant"blue")),
            layer(df,x=:iter,y=:obj3,Geom.line,Theme(default_color=colorant"orange")),
            layer(df,x=:iter,y=:obj4,Geom.line,Theme(default_color=colorant"green"))
            )
    img = SVG(filename, 6inch, 4inch)
    draw(img, p)
end

function plotArray(array,filename)
    df = DataFrame(iter = [identity(x) for x in 1:size(array,1)], obj = array)
    p = plot(layer(df,x=:iter,y=:obj,Geom.line,Theme(default_color=colorant"red")))
    img = SVG(filename, 6inch, 4inch)
    draw(img, p)
end

function plotLogReg(X)
    p = plot(iris, x=:X, y=:y, Geom.line)
    img = SVG("iris_plot.svg", 6inch, 4inch)
    draw(img, p)
end

end
