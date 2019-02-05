# draw Gaussian on circle

using Distributions
# using Fontconfig
using Cairo
using Gadfly, Colors
using Random

using ApproxManifoldProducts
using KernelDensityEstimate

using TransformUtils

const TU = TransformUtils


setForceEvalDirect!(true)

##


pd = Normal(-pi, 0.25)
pd2 = Normal(0.0, 0.75)
pd3 = MixtureModel([Normal(1.0, 0.3);Normal(-1.0, 0.2)], [0.6;0.4])

arr = Function[
  (x)->pdf(pd, x);
  (x)->pdf(pd2, x);
  (x)->pdf(pd3, x)
]


pl = plotCircBeliefs(arr)


## look at known failures


pts = [-pi/2; pi/2+1e-10]
pd3 = MixtureModel([Normal(pts[1], 0.1);Normal(pts[2], 0.1)], [0.5;0.5])


arr = Function[
  (x)->pdf(pd3, x);
]


pl = plotCircBeliefs(arr)


# pl |> PDF("/tmp/test.pdf")
# @async run(`evince /tmp/test.pdf`)




## Plot a KDE on circle


pts = rand(Uniform(-pi,pi), 100)

pc = kde!(pts, [0.1;])

gg = (x)->pc([x;])[1]

arr = [gg;]

pl = plotCircBeliefs(arr)



## plot two modes on circular manifold (using tree based eval)


# these do work
pts = [-pi/2; pi/2]
pts = [-pi/2; pi/2-1e-1]
pts = TU.wrapRad.([-pi/2; pi+0.1])
pts = [-pi/2+1e-1; pi/2]


# these dont work with difftheta, but fine with `-`
pts = [-pi/2; pi/2+1e-10]
pts = [-pi/2; pi/2+1.0]
pts = [-pi/2-1e-10; pi/2]




p = kde!(pts, [0.2], (addtheta,), (difftheta,))
# p.bt.centers



# circular plot does not work
plotKDECircular(p)




# linear plot works with similar eval
x=reshape(collect(range(-pi,stop=pi,length=1000)), 1,:)


# should work, but definitely doesn't
y = p( x, false, 1e-3, (addtheta,),(difftheta,) )
Gadfly.plot(x=x, y=y, Geom.line)


# shouldnt necessarily work, but does
# y = p( x, false, 1e-3, (+,),(-,) )
# Gadfly.plot(x=x, y=y, Geom.line)





gg = (x)->p([x;], false, 1e-3, (addtheta,), (difftheta,))[1]

arr = [gg;]

pl = plotCircBeliefs(arr)




## debugging the circular plotting function


arr = []

for bd in [p;]
  gg = (x::Float64)->bd([x;])[1]
  push!(arr, gg)
end

N=1000
th = range(-pi, stop=pi-1e-15, length=N)
c=["green"; "blue"; "deepskyblue"; "magenta"; "cyan"]
logpdf=true

c = cat(["red";], c, dims=1)
beliefs = Dict{Int, Function}()
beliefs[1] = (x)->1.0
global j = 1
for ar in arr
  global j
  j += 1
  beliefs[j] = logpdf ? (x)->log(ar(x)+1.0) : (x)->ar(x) #
end

PL = []
for j in 1:length(beliefs)
  obj = zeros(N)
  X = zeros(N)
  Y = zeros(N)
  for i in 1:N
    @show i, th
    @show t = th[i]
    obj[i] = 1.0 # beliefs[1](t) # = 1.0
    obj[i] += (j != 1 ? beliefs[j](t) : 0.0) # TODO: error must be here?
    # second term directional statistics for plotting (not evaluation)
    # obj[i] += (j != 1 ? beliefs[j](t-2pi) : 0.0 )
    # obj[i] += (j != 1 ? beliefs[j](t+2pi) : 0.0 )
    @show t, obj[i]
    X[i] = cos(t)*obj[i]
    Y[i] = sin(t)*obj[i]
  end
  push!(PL, Gadfly.layer(x=deepcopy(X), y=deepcopy(Y), Gadfly.Geom.path, Gadfly.Theme(default_color=parse(Colorant,c[j])))[1] )
end

Gadfly.plot(PL...)


# pts = [0.0001*randn(N2).-pi/2; 0.0001*randn(N2).+pi/2]
# shuffle!(pts)

## repeat with linear plot (still using same eval)


#
