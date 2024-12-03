
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

# DATADIR = joinpath(dirname(@__DIR__),"testdata")

##


@testset "Product of two Manellic beliefs, Sequential Gibbs, TranslationGroup(1)" begin
##

M = TranslationGroup(1)
N = 64

pts1 = [randn(1).-1 for _ in 1:N]
p1 = ApproxManifoldProducts.buildTree_Manellic!(M, pts1; kernel_bw=[0.1;;], kernel=ApproxManifoldProducts.MvNormalKernel)

pts2 = [randn(1).+1 for _ in 1:N]
p2 = ApproxManifoldProducts.buildTree_Manellic!(M, pts2; kernel_bw=[0.1;;], kernel=ApproxManifoldProducts.MvNormalKernel)

##

# tree kernel indices
@test 2 == ApproxManifoldProducts.leftIndex(p1, 1)
@test 3 == ApproxManifoldProducts.rightIndex(p1, 1)
# leaf kernel indices
@test N+1 == ApproxManifoldProducts.leftIndex(p1, floor(Int,N/2))
@test N+2 == ApproxManifoldProducts.rightIndex(p1, floor(Int,N/2))

@test ApproxManifoldProducts.exists_BTLabel(p1, floor(Int,N/2))
@test ApproxManifoldProducts.exists_BTLabel(p1, ApproxManifoldProducts.leftIndex(p1, floor(Int,N/2)))
@test !ApproxManifoldProducts.exists_BTLabel(p1, 2*N+1)


##

# leaves only in binary tree indexing
bt_label_pool = [
  [(N+1):(2*N);], # use leaf BT labels from p1 
  [(N+1):(2*N);], # use leaf BT labels from p2
]

# leaves only version
@info "Leaves only label sampling version (Gibbs), TranslationGroup(1)" 

ApproxManifoldProducts.sampleProductSeqGibbsBTLabel(M, [p1; p2], 3, bt_label_pool)

lbls = ApproxManifoldProducts.sampleProductSeqGibbsBTLabels(M, [p1; p2], 3, N, bt_label_pool)
post = ApproxManifoldProducts.calcProductKernelsBTLabels(M, [p1;p2], lbls, false) # ?? was permute=false?

pts = mean.(post)
kernel_bw = mean(cov.(post))
mtr = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw, kernel=ApproxManifoldProducts.MvNormalKernel)

@test isapprox( 0, mean(ApproxManifoldProducts.getKernelTree(mtr,1))[1]; atol=0.75)

@test all( (s->isapprox(1/N, s.weight;atol=1e-6)).(post) )


@info "Multi-scale label sampling version (Gibbs), TranslationGroup(1)" 

# test label pool creation
child_label_pools, all_leaves = ApproxManifoldProducts.generateLabelPoolRecursive([p1;p2], [1; 1])
@test !all_leaves
@test [2; 3] == child_label_pools[1]
@test [2; 3] == child_label_pools[2]

child_label_pools, all_leaves = ApproxManifoldProducts.generateLabelPoolRecursive([p1;p2], [floor(Int,N/2); 2*N])
@test !all_leaves
@test [N+1; N+2] == child_label_pools[1]
@test [2*N;] == child_label_pools[2]

child_label_pools, all_leaves = ApproxManifoldProducts.generateLabelPoolRecursive([p1;p2], [N+1; 2*N])
@test all_leaves
@test [N+1;] == child_label_pools[1]
@test [2*N;] == child_label_pools[2]


# test sampling
ApproxManifoldProducts.sampleProductSeqGibbsBTLabel(M, [p1; p2])

lbls = ApproxManifoldProducts.sampleProductSeqGibbsBTLabels(M, [p1; p2])
post = ApproxManifoldProducts.calcProductKernelsBTLabels(M, [p1;p2], lbls, false) # ?? was permute=false?

pts = mean.(post)
kernel_bw = mean(cov.(post))
mtr = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw, kernel=ApproxManifoldProducts.MvNormalKernel)

@test isapprox( 0, mean(ApproxManifoldProducts.getKernelTree(mtr,1))[1]; atol=0.75)


##
end


##

# using GLMakie


# XX = [[s;] for s in -4:0.1:4]
# YY = ApproxManifoldProducts.evaluate.(Ref(mtr), XX)

# lines((s->s[1]).(XX),YY, color=:magenta)

# YY = ApproxManifoldProducts.evaluate.(Ref(p1), XX)
# lines!((s->s[1]).(XX),YY, color=:blue)
# YY = ApproxManifoldProducts.evaluate.(Ref(p2), XX)
# lines!((s->s[1]).(XX),YY, color=:red)



@testset "Multi-scale label sampling version (Gibbs), TranslationGroup(2)" begin
##

M = TranslationGroup(2)
N = 64

pts1 = [1*randn(2) for _ in 1:N]
p1 = ApproxManifoldProducts.manikde!_manellic(M,pts1)

pts2 = [1*randn(2) for _ in 1:N]
p2 = ApproxManifoldProducts.manikde!_manellic(M,pts2)


# test sampling
lbls = ApproxManifoldProducts.sampleProductSeqGibbsBTLabels(M, [p1.belief; p2.belief])
lbls_ = unique(lbls)
N_ = length(lbls_)
weights = 1/N .* ones(N_)
# increase weight of duplicates
if N_ < N
  for (i,lb_) in enumerate(lbls_)
    idxs = findall(==(lb_),lbls)
    weights[i] = weights[i]*length(idxs)
  end
end
post = ApproxManifoldProducts.calcProductKernelsBTLabels(M, [p1.belief; p2.belief], lbls_, false; weights) # ?? was permute=false?
# check that any duplicates resulted in a height weight
@test isapprox( weights, (s->s.weight).(post); atol=1e-6 )

# NOTE, resulting tree might not have N number of data points 
mtr12 = ApproxManifoldProducts.buildTree_Manellic!(M,post)

##
end

