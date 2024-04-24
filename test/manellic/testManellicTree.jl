
# using Revise
using Test
using ApproxManifoldProducts
using Random
using StaticArrays
using TensorCast
using Manifolds
using Distributions
import ApproxManifoldProducts: ManellicTree, eigenCoords, splitPointsEigen

using Optim

using JSON3

##

DATADIR = joinpath(dirname(@__DIR__),"testdata")

# test 
function testEigenCoords(
  r_C = pi/3,
  ax_CC = [SA[5*randn();randn()] for _ in 1:100],
)
  M = TranslationGroup(2)
  _R(α, s=exp(-α*im)) = real(s)*SA[1 0; 0 1] + imag(s)*SA[0 1; -1 0]
  # _R(α) = SA[cos(α) sin(α); -sin(α) cos(α)]
  r_R_ax = _R(r_C)
  # rotate coordinates
  r_CC = map(ax_CC) do ax_C
    r_R_ax*ax_C + SA[10;-100]
  end
  r_CV = Manifolds.cov(M, r_CC)
  r_R_ax_, L, pidx = ApproxManifoldProducts.eigenCoords(r_CV)

  # spot check
  @show _ax_ERR = log_lie(SpecialOrthogonal(2), (r_R_ax_')*r_R_ax)[1,2]
  @show testval = isapprox(0, _ax_ERR; atol = 5/length(ax_CC))
  @assert testval "Spot check failed on eigen split of manifold points, the estimated point rotation matrix did not match construction."

  r_CC, r_R_ax_, pidx, r_CV
end

##
@testset "test ManellicTree construction" begin
##

M = TranslationGroup(2)
α = pi/3
r_CC, R, pidx, r_CV = testEigenCoords(α);
ax_CCp, mask, knl = splitPointsEigen(M, r_CC)
@test sum(mask) == (length(r_CC) ÷ 2)
@test knl isa ApproxManifoldProducts.MvNormalKernel
Mr = SpecialOrthogonal(2)
@test isapprox( α, vee(Mr, Identity(Mr), log_lie(Mr, R))[1] ; atol=0.1)

##

# using GLMakie
# fig = Figure()
# ax = Axis(fig[1,1])
# ptsl = ax_CCp[mask]
# ptsr = ax_CCp[xor.(mask,true)]
# plot!(ax, (s->s[1]).(ptsl), (s->s[2]).(ptsl), color=:blue)
# plot!(ax, (s->s[1]).(ptsr), (s->s[2]).(ptsr), color=:red)
# ax = Axis(fig[2,1])
# ptsl = r_CC[mask]
# ptsr = r_CC[xor.(mask,true)]
# plot!(ax, (s->s[1]).(ptsl), (s->s[2]).(ptsl), color=:blue)
# plot!(ax, (s->s[1]).(ptsr), (s->s[2]).(ptsr), color=:red)
# fig

## ensure that view of view can update original memory

A = randn(3)
A_ = view(A, 1:2)
A__ = view(A_, 1:1)
A__[1] = -100
@test isapprox(-100, A[1]; atol=1e-10)

##

r_PP = r_CC # shortcut because we are in Euclidean space
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, r_PP; kernel=AMP.MvNormalKernel)


##

@cast pts[i,d] := r_PP[i][d]

ptsl = pts[mtree.permute[1:50],:]
ptsr = pts[mtree.permute[51:100],:]

##

# fig = Figure()
# ax = Axis(fig[1,1])

# plot!(ax, ptsl[:,1], ptsl[:,2], color=:blue)
# plot!(ax, ptsr[:,1], ptsr[:,2], color=:red)

# fig

##

AMP.evaluate(mtree, SA[10.0;-101.0])


##
end


@testset "ManellicTree construction 1D" begin
##

M = TranslationGroup(1)
# already sorted list
pts = [[1.],[2.],[4.],[7.],[11.],[16.],[22.]]
bw = [1.0]
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)

@test  7 == length( intersect( mtree.segments[1], Set(1:7)) )
@test  4 == length( intersect( mtree.segments[2], Set(1:4)) )
@test  3 == length( intersect( mtree.segments[3], Set(5:7)) )
@test  2 == length( intersect( mtree.segments[4], Set(1:2)) )
@test  2 == length( intersect( mtree.segments[5], Set(3:4)) )
@test  2 == length( intersect( mtree.segments[6], Set(5:6)) )

@test isapprox( mean(M,pts), mean(mtree.tree_kernels[1]); atol=1e-6)
@test isapprox( mean(M,pts[1:4]), mean(mtree.tree_kernels[2]); atol=1e-6)
@test isapprox( mean(M,pts[5:7]), mean(mtree.tree_kernels[3]); atol=1e-6)
@test isapprox( mean(M,pts[1:2]), mean(mtree.tree_kernels[4]); atol=1e-6)
@test isapprox( mean(M,pts[3:4]), mean(mtree.tree_kernels[5]); atol=1e-6)
@test isapprox( mean(M,pts[5:6]), mean(mtree.tree_kernels[6]); atol=1e-6)


## additional test datasets

function testMDEConstr(
  pts::AbstractVector{<:AbstractVector{<:Real}},
  permref = sortperm(pts, by=s->getindex(s,1));
  lseg = 1:2,
  rseg = 3:4
)
  # check permutation
  M = TranslationGroup(1)
  bw = [1.0]

  mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)
  @test permref == mtree.permute
  @test isapprox( mean(M, pts), mean(mtree.tree_kernels[1]); atol=1e-10)
  @test Set(mtree.segments[1]) == Set(union(lseg, rseg))
  @test Set(mtree.segments[2]) == Set(mtree.permute[lseg])
  @test Set(mtree.segments[3]) == Set(mtree.permute[rseg])
  @test isapprox( mean(M, pts[mtree.permute[lseg]]), mean(mtree.tree_kernels[2]); atol=1e-10)
  @test isapprox( mean(M, pts[mtree.permute[rseg]]), mean(mtree.tree_kernels[3]); atol=1e-10)
  nothing
end

## for 4 values

# manual orders
testMDEConstr( [[0.;],[1.],[3.;],[6.;]] )
testMDEConstr( [[0.;],[1.],[6.;],[3.;]] )
testMDEConstr( [[0.;],[3.],[1.;],[6.;]] )
testMDEConstr( [[1.;],[0.],[3.;],[6.;]] )
testMDEConstr( [[1.;],[0.],[6.;],[3.;]] )
testMDEConstr( [[1.;],[6.],[0.;],[3.;]] )
testMDEConstr( [[6.;],[1.],[3.;],[0.;]] )

testMDEConstr( [[0.9497270480266986;], [-0.5973125859935883;], [-0.6031001429225558;], [-0.3971695179687664;]] )


# randomized orders for 4 values
pts = [[0.;],[1.],[3.;],[6.;]]
for i in 1:10
  testMDEConstr( pts[shuffle(1:4)] )
end


## for 5 values

testMDEConstr( [[0.;],[1.],[3.;],[6.;],[10.;]]; lseg=1:3, rseg=4:5)
testMDEConstr( [[0.;],[1.],[6.;],[3.;],[10.;]]; lseg=1:3, rseg=4:5)

# randomized orders for 5 values
pts = [[0.;],[1.],[3.;],[6.;],[10.;],]
for i in 1:10
  testMDEConstr( pts[shuffle(1:length(pts))], lseg=1:3, rseg=4:5)
end

## for 7 values

# randomized orders for 7 values
pts = [[0.;],[1.],[3.;],[6.;],[10.;],[15.;],[21.;]]
for i in 1:10
  testMDEConstr( pts[shuffle(1:length(pts))]; lseg=1:4,rseg=5:7 )
end


#
M = TranslationGroup(1)
pts = [randn(1) for _ in 1:8]
for i in 1:10
  _pts = pts[shuffle(1:length(pts))]
  testMDEConstr( _pts; lseg=1:4,rseg=5:8 )
end


##
end


@testset "ManellicTree 1D basic construction and evaluations" begin
## 

M = TranslationGroup(1)
pts = [randn(1) for _ in 1:128]
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel=AMP.MvNormalKernel)

AMP.evaluate(mtree, SA[0.0;])

## load know test data test

json_string = read(joinpath(DATADIR,"manellic_test_data.json"), String)
dict = JSON3.read(json_string, Dict{Symbol,Vector{Float64}})

M = TranslationGroup(1)
pts = [[v;] for v in dict[:evaltest_1_pts]]
bw = reshape(dict[:evaltest_1_bw],1,1)
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)

np = Normal(0,1)
h = 0.1
xx = -5:h:5
yy_ = pdf.(np, xx) # ref
yy = [AMP.evaluate(mtree, [v;]) for v in xx] # test
for (i,v) in enumerate(yy_)
  @test isapprox(v, yy[i]; atol=0.05)
end
@test isapprox( 1, sum(yy_ .* h) ; atol=1e-3)
@test isapprox( 1, sum(yy .* h) ; atol=1e-3)
# using GLMakie
# lines(xx, yy_, color=:red) # ref
# lines!(xx, yy, color=:blue) # test

# test sorting order of data 
permref = sortperm(pts, by=s->getindex(s,1))

@test 0 == sum(permref - mtree.permute)

@test 0 == sum(collect(sortperm(mtree.leaf_kernels; by=s->mean(s))) - collect(1:length(mtree.leaf_kernels)))

#and leaf kernel sorting
@test norm((pts[mtree.permute] .- mean.(mtree.leaf_kernels)) .|> s->s[1]) < 1e-6



# for (i,v) in enumerate(dict[:evaltest_1_at])
#   # @show AMP.evaluate(mtree, [v;]), dict[:evaltest_1_dens][i]
#   @test isapprox(dict[:evaltest_1_dens][i], AMP.evaluate(mtree, [v;]))
# end
# isapprox(dict[:evaltest_1_dens][5], AMP.evaluate(mtree, [dict[:evaltest_1_at][5]]))
# eval test ref Normal(0,1)

##
end

@testset "Manellic basic evaluation test 1D" begin
##

M = TranslationGroup(1)
pts = [zeros(1) for _ in 1:100]
bw = ones(1,1)
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw, kernel=AMP.MvNormalKernel)

@test isapprox( 0.4, AMP.evaluate(mtree, SA[0.0;]); atol=0.1)

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


##
end


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

# and optimize

res = Optim.optimize(
  (s)->cost(pts,s^2), 
  0.05, 3.0, Optim.GoldenSection()
)
best_cov = Optim.minimizer(res)

@test isapprox(0.5, best_cov; atol=0.3)


##
end



# TODO
# @testset "Manellic tree bandwidth optimize n-dim RLM" begin
# ##


# ##
# end



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


@testset "Test utility functions for Gaussian products" begin
##

M = TranslationGroup(1)

g1 = ApproxManifoldProducts.MvNormalKernel([-1.0;],[4.0;;])
g2 = ApproxManifoldProducts.MvNormalKernel([1.0;],[4.0;;])

g = ApproxManifoldProducts.calcProductGaussians(M, [g1; g2])
@test isapprox( [0.0;], mean(g); atol=1e-6)
@test isapprox( [2.0;;], cov(g); atol=1e-6)

g1 = ApproxManifoldProducts.MvNormalKernel([-1.0;],[4.0;;])
g2 = ApproxManifoldProducts.MvNormalKernel([1.0;],[9.0;;])

g = ApproxManifoldProducts.calcProductGaussians(M, [g1; g2])
@test isapprox( [-5/13;], mean(g); atol=1e-6)
@test isapprox( [36/13;;], cov(g); atol=1e-6)

##
end


# @testset "Test utility functions for multi-scale product sampling" begin
# ##

# M = TranslationGroup(1)

# pts = [randn(1).-1 for _ in 1:3]
# p1 = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=[0.1;;], kernel=ApproxManifoldProducts.MvNormalKernel)

# @test 1 == length(ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 1))
# @test 2 == length(ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 2))
# @test 4 == length(ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 3))

# @test 64 == length(ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 7))
# @test 128 == length(ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 8))

# # @enter 
# ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 2)
# ApproxManifoldProducts.getKernelsTreeLevelIdxs(p1, 3)



# ##
# end


@testset "Product of two Manellic beliefs, Sequential Gibbs" begin
##

M = TranslationGroup(1)
N = 4

pts1 = [randn(1).-2 for _ in 1:N]
p1 = ApproxManifoldProducts.buildTree_Manellic!(M, pts1; kernel_bw=[0.1;;], kernel=ApproxManifoldProducts.MvNormalKernel)

pts2 = [randn(1).+2 for _ in 1:N]
p2 = ApproxManifoldProducts.buildTree_Manellic!(M, pts2; kernel_bw=[0.1;;], kernel=ApproxManifoldProducts.MvNormalKernel)

##



##

# leaves only
label_sets = [
  [(N+1):(2*N);], # use leaf BT labels from p1 
  [(N+1):(2*N);], # use leaf BT labels from p2
]

# leaves only version
lbls = ApproxManifoldProducts.sampleProductSeqGibbsBTLabels(M, [p1; p2], 3, N, label_sets)

post = ApproxManifoldProducts.calcProductKernelsBTLabels(M, [p1;p2], lbls) # ?? was permute=false?

pts = mean.(post)
kernel_bw = mean(cov.(post))

mtr = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw, kernel=ApproxManifoldProducts.MvNormalKernel)

@error "DEV SKIP OF TEST, MUST RESTORE"
# @test isapprox( 0, mean(ApproxManifoldProducts.getKernelTree(mtr,1))[1]; atol=0.75)


# @info "multi-scale sample test" 

# lbls = ApproxManifoldProducts.sampleProductSeqGibbsBTLabels(M, [p1; p2])


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


#