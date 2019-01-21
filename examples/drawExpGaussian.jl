# draw Gaussian on circle

using Distributions
# using Fontconfig
using Cairo
using Gadfly, Colors

using ApproxManifoldProducts




pd = Normal(-pi, 0.5)
pd2 = Normal(0.6*pi, 0.75)
pd3 = MixtureModel([Normal(1.0, 0.3);Normal(-1.0, 0.2)], [0.6;0.4])

arr = Function[
  (x)->pdf(pd, x);
  (x)->pdf(pd2, x);
  (x)->pdf(pd3, x)
]


pl = plotCircBeliefs(arr)






pl |> PDF("/tmp/test.pdf")
@async run(`evince /tmp/test.pdf`)




## Plot a KDE on circle

using Distributions

pts = rand(Uniform(-pi,pi), 100)

pc = kde!(pts, [0.1;])

gg = (x)->pc([x;])[1]

arr = [gg;]

pl = plotCircBeliefs(arr)

#
