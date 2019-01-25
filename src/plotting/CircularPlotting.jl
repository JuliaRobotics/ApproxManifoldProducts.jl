# plotting functions of manifold beliefs

using .Gadfly
using .Colors

export plotCircBeliefs, plotKDECircular


function plotCircBeliefs(arr::V;
                         N::Int=1000,
                         th = range(-pi, stop=pi-1e-5, length=N),
                         c=["green"; "blue"; "deepskyblue"; "magenta"; "cyan"],
                         logpdf::Bool=true  ) where {V <: Vector}
  #
  c = ["red";c]
  beliefs = Dict{Int, Function}()
  beliefs[1] = (x)->1.0
  j = 1
  for ar in arr
    j += 1
    beliefs[j] = logpdf ? (x)->log(ar(x)+1.0) : (x)->ar(x) #
  end

  PL = []
  for j in 1:length(beliefs)
    obj = zeros(N)
    X = zeros(N)
    Y = zeros(N)
    for i in 1:N
      t = th[i]
      obj[i] = beliefs[1](t)
      obj[i] += (j != 1 ? beliefs[j](t) : 0.0)
      obj[i] += (j != 1 ? beliefs[j](t-2pi) : 0.0 )
      obj[i] += (j != 1 ? beliefs[j](t+2pi) : 0.0 )
      X[i] = cos(t)*obj[i]
      Y[i] = sin(t)*obj[i]
    end
    push!(PL, Gadfly.layer(x=deepcopy(X), y=deepcopy(Y), Gadfly.Geom.path, Gadfly.Theme(default_color=parse(Colorant,c[j])))[1] )
  end

  Gadfly.plot(PL...)
end

function plotKDECircular(bds::Vector{BallTreeDensity};
                         c=["green"; "blue"; "deepskyblue"; "magenta"; "cyan"],
                         logpdf::Bool=true  )

  arr = []

  for bd in bds
    gg = (x::Float64)->bd([x;])[1]
    push!(arr, gg)
  end

  return plotCircBeliefs(arr, c=c, logpdf=logpdf)
end

function plotKDECircular(bd::BallTreeDensity; c=["green";], logpdf::Bool=true)
  return plotKDECircular([bd;], c=c, logpdf=logpdf)
end
