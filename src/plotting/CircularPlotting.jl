# plotting functions of manifold beliefs

using .Gadfly
using .Colors

export plotCircBeliefs



function plotCircBeliefs(arr::Vector;
                         N::Int=1000,
                         th = range(-pi, stop=pi-1e-5, length=N) )
  #
  beliefs = Dict{Int, Function}()
  beliefs[1] = (x)->1.0
  j = 1
  for ar in arr
    j += 1
    beliefs[j] = ar #
  end

  CL = [colorant"red"; colorant"green"; colorant"blue"; colorant"deepskyblue"; colorant"magenta"; colorant"cyan"]
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
    push!(PL, Gadfly.layer(x=deepcopy(X), y=deepcopy(Y), Gadfly.Geom.path, Gadfly.Theme(default_color=CL[j]))[1] )
  end

  Gadfly.plot(PL...)
end
