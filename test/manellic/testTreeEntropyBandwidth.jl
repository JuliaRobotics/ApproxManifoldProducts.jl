

# using Revise
using Test
using ApproxManifoldProducts
using Random
using LinearAlgebra
using StaticArrays
using TensorCast
using Manifolds
import Rotations as Rot_
using Distributions
import ApproxManifoldProducts: ManellicTree, eigenCoords, splitPointsEigen

using Optim

using JSON3


##


@testset "Manellic tree kernel bandwidth 1D LOO evaluation/entropy checks" begin
##

M = TranslationGroup(1)

pt = [[0.0;],[1.0;],[2.0;],[3.0;]]
_m_ = ApproxManifoldProducts.buildTree_Manellic!(M, pt; kernel_bw=[1.0;;],kernel=AMP.MvNormalKernel)
ApproxManifoldProducts.entropy(_m_)

XX = -3:0.1:6
Y = [ApproxManifoldProducts.evaluate(_m_, [x;]) for x in XX]

function cost(s)
  mtr = ApproxManifoldProducts.buildTree_Manellic!(M, pt; kernel_bw=[s;;],kernel=AMP.MvNormalKernel)
  # AMP.entropy(mtr)
  AMP.expectedLogL(mtr, getPoints(mtr), true)
end

# optimal is somewhere in the single digits and basic monoticity outward
@test cost(1e-4) === -Inf
@test cost(1e-3) < cost(1e-2) < cost(1e-1) < cost(1e-0)
@test cost(1e2) < cost(1e1) < cost(1e0)

# S = [1e-3; 1e-2; 1e-1; 1e0; 1e1; 1e2]
# Y = cost.(S)

##
end


@testset "Manellic tree bandwidth optimization 1D section search" begin
##

M = TranslationGroup(1)
# pts = [[0.;],[0.1],[0.2;],[0.3;]]
pts = [1*randn(1) for _ in 1:64]

bw = [1.0]
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)
# TODO isdefined does not work here (upstream bug somewhere)
# @test isdefined(mtree.tree_kernels, 1)
# @test isdefined(mtree.tree_kernels, 2)
# @test isdefined(mtree.tree_kernels, 3)
# @test !isdefined(mtree.tree_kernels, 4)


## ASSUMING SCALAR
# do linesearch for best selection of bw_scl
# MINIMIZE(entropy, mtree, p0)

# FIXME use bounds
lcov, ucov = AMP.getBandwidthSearchBounds(mtree)
bw_cov = (ucov + lcov)/2
mtree_0 = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw_cov,kernel=AMP.MvNormalKernel)
lower = lcov / bw_cov
upper = ucov / bw_cov
AMP.entropy(mtree_0)



# https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/#minimizing-a-univariate-function-on-a-bounded-interval
# options for kwargs...
# iterations
# rel_tol: The relative tolerance used for determining convergence. Defaults to sqrt(eps(T))
# abs_tol: The absolute tolerance used for determining convergence. Defaults to eps(T)
cost(_pts, σ) = begin
  mtr = ApproxManifoldProducts.buildTree_Manellic!(M, _pts; kernel_bw=[σ;;],kernel=AMP.MvNormalKernel)
  AMP.entropy(mtr)
end


S = 0.005:0.05:3
Y = S .|> s->cost(pts,s^2)

# should pass the optimal kbw somewhere in the given range
@test any(0 .< diff(Y))

# and optimize with rebuild tree cost
res = Optim.optimize(
  (s)->cost(pts,s^2), 
  0.05, 3.0, Optim.GoldenSection()
)

best_cov = Optim.minimizer(res)

@test isapprox(0.5, best_cov; atol=0.3)
bcov_ = deepcopy(best_cov)

## Test more efficient updateKernelBW version

cost2(σ) = begin
  mtr = ApproxManifoldProducts.updateBandwidths(mtree_0, [σ;;])
  AMP.entropy(mtr)
end

# and optimize with "update" kernel bandwith cost
res = Optim.optimize(
  (s)->cost2(s^2), 
  0.05, 3.0, Optim.GoldenSection()
)

@show best_cov = Optim.minimizer(res)

@test isapprox(bcov_, best_cov; atol=1e-3)

# mask bandwith by passing in an alternative

cost3(σ) = begin
  AMP.entropy(mtree_0, [σ;;])
end

# and optimize with "update" kernel bandwith cost
res = Optim.optimize(
  (s)->cost3(s^2), 
  0.05, 3.0, Optim.GoldenSection()
)

best_cov = Optim.minimizer(res)

@test isapprox(bcov_, best_cov; atol=1e-3)


##
end



# TODO
@testset "Manellic tree all up construction with bandwith optimization" begin
##


M = TranslationGroup(1)
# pts = [[0.;],[0.1],[0.2;],[0.3;]]
pts = [1*randn(1) for _ in 1:128]

mkd = ApproxManifoldProducts.manikde!_manellic(M,pts)

best_cov = cov(ApproxManifoldProducts.getKernelLeaf(mkd.belief,1))[1] |> sqrt
@show best_cov

@test isapprox(0.5, best_cov; atol=0.3)

# remember broken code in get w bounds

try
  pts = [1*randn(1) for _ in 1:100]
  mkd = ApproxManifoldProducts.manikde!_manellic(M,pts)
catch
  @test_broken false
end


##
end


@testset "Multidimensional LOOCV bandwidth optimization, TranslationGroup(2)" begin
##

M = TranslationGroup(2)
pts = [1*randn(2) for _ in 1:64]

bw = [1.0; 1.0]
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)

cost4(σ) = begin
  AMP.entropy(mtree, diagm(σ.^2))
end

# and optimize with "update" kernel bandwith cost
@time res = Optim.optimize(
  cost4, 
  bw, 
  Optim.NelderMead()
);

@test res.ls_success

@show best_cov = abs.(Optim.minimizer(res))

@test isapprox([0.5; 0.5], best_cov; atol=0.3)


mkd = ApproxManifoldProducts.manikde!_manellic(M,pts)

@test isapprox([0.5 0; 0 0.5], getBW(mkd)[1]; atol=0.3)


##
end



@testset "Multidimensional LOOCV bandwidth optimization, SpecialEuclidean(2; vectors=HybridTangentRepresentation())" begin
##

M = SpecialEuclidean(2; vectors=HybridTangentRepresentation())
pts = [ArrayPartition(randn(2),Rot_.RotMatrix{2}(0.1*randn()).mat) for _ in 1:64]

bw = [1.0; 1.0; 0.3]
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)

cost4(σ) = begin
  AMP.entropy(mtree, diagm(σ.^2))
end

# and optimize with "update" kernel bandwith cost
@time res = Optim.optimize(
  cost4, 
  bw, 
  Optim.NelderMead()
);

@test res.ls_success

@show best_cov = abs.(Optim.minimizer(res))

@test isapprox([0.6; 0.6], best_cov[1:2]; atol=0.35)
@test isapprox(0.06, best_cov[3]; atol=0.04)


mkd = ApproxManifoldProducts.manikde!_manellic(M,pts)

@test isapprox([0.6 0; 0 0.6], getBW(mkd)[1][1:2,1:2]; atol=0.4)
@test isapprox(0.06, getBW(mkd)[1][3,3]; atol=0.04)


##
end



@testset "Multidimensional LOOCV bandwidth optimization, SpecialEuclidean(3; vectors=HybridTangentRepresentation())" begin
##

M = SpecialEuclidean(3; vectors=HybridTangentRepresentation())
pts = [ArrayPartition(SA[randn(3)...;],SMatrix{3,3,Float64}(collect(Rot_.RotXYZ(0.1*randn(3)...)))) for _ in 1:64]

bw = SA[1.0; 1.0; 1.0; 0.3; 0.3; 0.3]
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)

cost4(σ) = begin
  AMP.entropy(mtree, diagm(σ.^2))
end

# and optimize with "update" kernel bandwith cost
@time res = Optim.optimize(
  cost4, 
  collect(bw), 
  Optim.NelderMead()
);

@test res.ls_success

@show best_cov = abs.(Optim.minimizer(res))

@test isapprox([0.75; 0.75; 0.75], best_cov[1:3]; atol=0.55)
@test isapprox([0.06; 0.06; 0.06], best_cov[4:6]; atol=0.055)


mkd = ApproxManifoldProducts.manikde!_manellic(M,pts)

@test isapprox([0.75 0 0; 0 0.75 0; 0 0 0.75], getBW(mkd)[1][1:3,1:3]; atol=0.55)
@test isapprox([0.07 0 0; 0 0.07 0; 0 0 0.07], getBW(mkd)[1][4:6,4:6]; atol=0.055)

##
end

##
# # using GLMakie

# M = TranslationGroup(1)

# __pts = [1*randn(1) for _ in 1:64] 


# ##

# f = Figure()
# ax = f[1, 1] = Axis(f; xscale=log10,yscale=log10)

# ##

# SFT = 0.0
# pts = [p .+ SFT for p in __pts]

# ##

# S = 0.005:0.01:2
# Y = S .|> s->cost(pts,s^2)

# lines!(S, Y .+ 0.1, color=:blue, label="Manellic $SFT")

# f[1, 2] = Legend(f, ax, "Entropy R&D", framevisible = false)
# f

##
