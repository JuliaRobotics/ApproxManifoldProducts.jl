

using ApproxManifoldProducts
# using Distributions


##




# these do work
ptsA = [-pi/2; pi/2]

# these dont work with difftheta, but fine with `-`
ptsB = [-pi/2; pi/2+1e-10]



pA = kde!(ptsA, [0.1], (addtheta,), (difftheta,))
pB = kde!(ptsB, [0.1], (addtheta,), (difftheta,))


##


ept = ones(1,1)
ept[1,1] = pi/2

resA = evaluateDualTree(pA, ept, false, 1e-3, (addtheta,), (difftheta,))[1]

println("")
println("")

resB = evaluateDualTree(pB, ept, false, 1e-3, (addtheta,), (difftheta,))[1]



##


pA.bt.centers





pB.bt.centers





#
