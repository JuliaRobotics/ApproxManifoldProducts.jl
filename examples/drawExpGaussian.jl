# draw Gaussian on circle

using Distributions
using Fontconfig
using Cairo
using Gadfly
using Colors



function plotCircBeliefs(arr::Vector{Function};N=1000, th = range(-pi, stop=pi-1e-5, length=N) )

  beliefs = Dict{Int, Function}()
  beliefs[1] = (x)->1.0
  j = 1
  for ar in arr
    j += 1
    beliefs[j] = ar #
  end

  CL = [colorant"red"; colorant"green"; colorant"blue"; colorant"deepskyblue"]
  PL = []
  for j in 1:length(beliefs)
    obj = zeros(N)
    X = zeros(N)
    Y = zeros(N)
    for i in 1:N
      t = th[i]
      obj[i] = beliefs[j](t) + (j != 1 ? beliefs[1](t) : 0.0)
      X[i] = cos(t)*obj[i]
      Y[i] = sin(t)*obj[i]
    end
    push!(PL, layer(x=deepcopy(X), y=deepcopy(Y), Geom.path, Theme(default_color=CL[j]))[1] )
  end

  plot(PL...)
end


pd = Normal(0.0, 0.5)
pd2 = Normal(pi/2, 0.75)

arr = Function[
  (x)->pdf(pd, x);
  (x)->pdf(pd2, x)
]



pl = plotCircBeliefs(arr)




draw(PDF("/tmp/test.pdf", 20cm, 15cm), pl)
@async run(`evince /tmp/test.pdf`)

#
