

using LinearAlgebra
using Random
using Distributions

# plot features require Gadfly, etc
using ApproxManifoldProducts

# more packages required
using TransformUtils


const LinAlg = LinearAlgebra
const KDE = KernelDensityEstimate
const TU = TransformUtils
const AMP = ApproxManifoldProducts



# temporarily required for experimental versions of on-manifold products
KDE.setForceEvalDirect!(true)



## more standard interface

@testset "test standard API for AMP.manikde!" begin

# two densities on a cylinder
p = manikde!(randn(2,100), (:Euclid, :Circular) )

pts2a = 3.0*randn(1,100).+5.0
pts2b = TransformUtils.wrapRad.(0.5*randn(1,100).+pi)
q = manikde!([pts2a;pts2b], (:Euclid, :Circular) )

# approximate the product between hybrid manifold densities
pq = manifoldProduct([p;q], (:Euclid, :Circular), Niter=2)



## check marginals

@test getBW( marginal(pq, [1;]) )[1,1] == getBW(pq)[1,1]
@test getBW( marginal(pq, [2;]) )[1,1] == getBW(pq)[2,1]


@test 0 < getBW(pq)[1,1] < 10.0
@test 0 < getBW(pq)[2,1] < 1.5


end



## lower level API



@testset "test lower level API..." begin


pts1a = 180.0.+10.0*randn(100)
pts2a = 200.0.+10.0*randn(100)

pts1b = TU.wrapRad.(0.1*randn(100).-pi.+0.5)
pts2b = TU.wrapRad.(0.1*randn(100).+pi.-0.5)

pts1 = [pts1a';pts1b']
pts2 = [pts2a';pts2b']

# pc1 = kde!_CircularNaiveCV(pts1)
# pc2 = kde!_CircularNaiveCV(pts2)
pc1 = kde!(pts1, [3.0; 0.1], (+,AMP.addtheta), (-,AMP.difftheta))
pc2 = kde!(pts2, [3.0; 0.1], (+,AMP.addtheta), (-,AMP.difftheta))




dummy = kde!(rand(2,100),[1.0;], (+,AMP.addtheta), (-,AMP.difftheta));

pGM, = prodAppxMSGibbsS(dummy, [pc1; pc2], nothing, nothing, Niter=1,
                          addop=(+,AMP.addtheta), diffop=(-,AMP.difftheta), getMu=(KDE.getEuclidMu,AMP.getCircMu));
#



lin1 = getBW(kde!(pGM[1,:]))[:,1][1]
cir1 = getBW(kde!_CircularNaiveCV(pGM[2,:]))[:,1][1]


pc12 = kde!(pGM, [lin1;cir1], (+,AMP.addtheta), (-,AMP.difftheta))


end
