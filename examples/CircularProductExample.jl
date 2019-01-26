# Test naive implementation of entropy calculations towards efficient calculation of entropy on a manifold

using DocStringExtensions
using KernelDensityEstimate

include(joinpath(dirname(@__FILE__), "circularEntropyUtils.jl"))

using Gadfly, Colors
using Distributions
using Random

using TransformUtils
using Optim

using ApproxManifoldProducts


const TU = TransformUtils



# On-manifold circular product callbacks

# manifold distance, add, and subtract
function logmap_SO2(Rl::Matrix{Float64})
  ct = abs(Rl[1,1]) > 1.0 ? 1.0 : Rl[1,1]  # reinserting the sign below
  sign(Rl[2,1])*acos(ct)
end
difftheta(wth1::Float64, wth2::Float64)::Float64 = logmap_SO2(TU.R(wth1)'*TU.R(wth2))
addtheta(wth1::Float64, wth2::Float64) = TU.wrapRad( wth1+wth2 )

# manifold get Gaussian products mean

getCircMu = (m::Vector{Float64}, s::Vector{Float64}, dummy::Float64) -> TU.wrapRad(get2DMuMin(m, s, diffop=difftheta, initrange=(-pi+0.0,pi+0.0)))



# create two densities

pts1 = TU.wrapRad.(0.1*randn(100).-pi.+0.5)
pts2 = TU.wrapRad.(0.1*randn(100).+pi.-0.5)

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.1;])
pc2 = kde!(pts2, [0.1;])


##

plotKDECircular([pc1;pc2])



## Calculate the approximate product between the densities


# TODO: make circular KDE
dummy = kde!(rand(100),[1.0;]);

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=1, addop=addtheta, diffop=difftheta, getMu=getCircMu);
pGM



pc12 = kde!_CircularNaiveCV(pGM[:])
getBW(pc12)[1,1]
pc12 = kde!(pGM, [0.1;])

pl = plotKDECircular([pc1;pc2; pc12])


##


Gadfly.push_theme(:default)



using Cairo
pl |> PDF("/tmp/test.pdf", 20cm,15cm)



## handrolically trying product of 3




pts1 = TU.wrapRad.(0.1*randn(100))
pts2 = TU.wrapRad.(0.1*randn(100).-2pi/3.0)
pts3 = TU.wrapRad.(0.1*randn(100).+2pi/3.0)


# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [0.5;])
pc2 = kde!(pts2, [0.1;])
pc3 = kde!(pts3, [0.1;])


pl = plotKDECircular([pc1;pc2;pc3])


dummy = kde!(rand(100),[1.0;]);

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=1, addop=addtheta, diffop=difftheta, getMu=getCircMu);
pGM



pc123 = kde!_CircularNaiveCV(pGM[:])



pl = plotKDECircular([pc1;pc2;pc3; pc123])



#
