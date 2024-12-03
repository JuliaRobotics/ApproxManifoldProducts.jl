
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

DATADIR = joinpath(dirname(@__DIR__),"testdata")


##



@testset "Manellic basic evaluation test 1D" begin
##

M = TranslationGroup(1)
pts = [zeros(1) for _ in 1:100]
bw = ones(1,1)
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw, kernel=AMP.MvNormalKernel)

@test isapprox( pdf(Normal(0,1), 0), AMP.evaluate(mtree, SA[0.0;]))

@error "expectedLogL for different number of test points not working yet."
# AMP.expectedLogL(mtree, [randn(1) for _ in 1:5])

@show AMP.entropy(mtree)

# Vector bw required for backward compat with legacy belief structure
mtreeV = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=[1.0;], kernel=AMP.MvNormalKernel)


bel = manikde!(
    M,
    pts;
    bw,
    belmodel = (a,b,aF,dF) -> ApproxManifoldProducts.buildTree_Manellic!(
    M,
    pts;
    kernel_bw=b, 
    kernel=AMP.MvNormalKernel
    )
)


@test isapprox( 0.4, bel([0.0;]); atol=0.1)

##
end


@testset "Manellic tree bandwidth evaluation" begin
## load know test data test

json_string = read(joinpath(DATADIR,"manellic_test_data.json"), String)
dict = JSON3.read(json_string, Dict{Symbol,Vector{Float64}})

M = TranslationGroup(1)
pts = [[v;] for v in dict[:evaltest_1_pts]]
bw = reshape(dict[:evaltest_1_bw],1,1)
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)

AMP.expectedLogL(mtree, pts)

@test AMP.expectedLogL(mtree, pts) < Inf 

# to enable faster bandwidth selection/optimization
ekr = ApproxManifoldProducts.getKernelLeaf(mtree,1,false)
ekr_ = ApproxManifoldProducts.updateKernelBW(ekr,SA[1.0;;])

@test typeof(ekr) == typeof(ekr_)

# confirm that updating the bandwidths works properly
Σ = [0.1+0.5*rand();;]

mtr = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=Σ,kernel=AMP.MvNormalKernel)
mtr_ = ApproxManifoldProducts.updateBandwidths(mtree, Σ)

# 
@test isapprox( mtr([0.0]), mtr_([0.0]); atol=1e-10)
@test isapprox( ApproxManifoldProducts.entropy(mtr), ApproxManifoldProducts.entropy(mtr_); atol=1e-10)


##
end
    