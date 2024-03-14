
# using Revise
using Test
using ApproxManifoldProducts
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
@test knl isa MvNormal
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
pts = [randn(1) for _ in 1:100]
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel=AMP.MvNormalKernel)

AMP.evaluate(mtree, SA[0.0;])

## load know test data test

json_string = read(joinpath(DATADIR,"manellic_test_data.json"), String)
dict = JSON3.read(json_string, Dict{Symbol,Vector{Float64}})

M = TranslationGroup(1)
pts = [[v;] for v in dict[:evaltest_1_pts]]
bw = reshape(dict[:evaltest_1_bw],1,1)
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)

# for (i,v) in enumerate(dict[:evaltest_1_at])
#   # @show AMP.evaluate(mtree, [v;]), dict[:evaltest_1_dens][i]
#   @test isapprox(dict[:evaltest_1_dens][i], AMP.evaluate(mtree, [v;]))
# end
# isapprox(dict[:evaltest_1_dens][5], AMP.evaluate(mtree, [dict[:evaltest_1_at][5]]))
# eval test ref Normal(0,1)

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


# check permutation
M = TranslationGroup(1)
pts = [[0.;],[1.],[2.;],[3.;]]
bw = [1.0]
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)

# TODO untested
@test mtree.permute == [1;2;3;4]

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


#

@testset "Manellic tree bandwidth evaluation" begin
## load know test data test

json_string = read(joinpath(DATADIR,"manellic_test_data.json"), String)
dict = JSON3.read(json_string, Dict{Symbol,Vector{Float64}})

M = TranslationGroup(1)
pts = [[v;] for v in dict[:evaltest_1_pts]]
bw = reshape(dict[:evaltest_1_bw],1,1)
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw,kernel=AMP.MvNormalKernel)

AMP.expectedLogL(mtree, pts)

@test AMP.expectedLogL(mtree, pts, 1.1) < AMP.expectedLogL(mtree, pts, 1.0) < AMP.expectedLogL(mtree, pts, 0.9)


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
  AMP.expectedLogL(mtr, getPoints(mtr), 1, true)
end

# optimal is somewhere in the single digits and basic monoticity outward
@test cost(1e-4) === -Inf
@test cost(1e-3) < cost(1e-2) < cost(1e-1) < cost(1e-0)
@test cost(1e2) < cost(1e1) < cost(1e0)


##
end


@testset "Manellic tree bandwidth optimization 1D section search" begin
##

M = TranslationGroup(1)
# pts = [[0.;],[1.],[2.;],[3.;]]
pts = [randn(1) for _ in 1:128]

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
AMP.entropy(mtree_0, lower[1])
AMP.entropy(mtree_0, upper[1])


# https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/#minimizing-a-univariate-function-on-a-bounded-interval
# options for kwargs...
# iterations
# rel_tol: The relative tolerance used for determining convergence. Defaults to sqrt(eps(T))
# abs_tol: The absolute tolerance used for determining convergence. Defaults to eps(T)
cost(s) = begin
  mtr = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=[s;;],kernel=AMP.MvNormalKernel)
  AMP.entropy(mtr)
end

res = Optim.optimize(
  cost, 
  0.05, 0.8, Optim.GoldenSection()
)
best_cov = Optim.minimizer(res)

@test_broken isapprox(0.38, best_cov; atol=0.15)


# test why broken

function cost(s)
  mtr = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=[s;;],kernel=AMP.MvNormalKernel)
  # AMP.entropy(mtr)
  AMP.expectedLogL(mtr, getPoints(mtr), 1, true)
end


XX = 0.05:0.05:0.3
YY = cost.(XX)

# should pass the optimal kbw somewhere in the given range
@test_broken any(0 .< diff(YY))


##
end



##
# using GLMakie

# f = Figure()

# ax = f[1, 1] = Axis(f; xscale=log10,yscale=log10)

# lines!(S, -Y,  color=:blue, label="Manellic")

# f[1, 2] = Legend(f, ax, "Entropy R&D", framevisible = false)

# f

##

# TODO
# @testset "Manellic tree bandwidth optimize n-dim RLM" begin
# ##


# ##
# end

#