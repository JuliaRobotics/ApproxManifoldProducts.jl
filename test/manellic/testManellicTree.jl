
# using Revise
using Test
using ApproxManifoldProducts
using StaticArrays
using TensorCast
using Manifolds
using Distributions
import ApproxManifoldProducts: ManellicTree, eigenCoords, splitPointsEigen

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


##
end

@testset "Manellic basic evaluation test 1D" begin
##

M = TranslationGroup(1)
pts = [zeros(1) for _ in 1:100]
bw = ones(1,1)
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw=bw, kernel=AMP.MvNormalKernel)

AMP.evaluate(mtree, SA[0.0;])

AMP.evalAvgLogL(mtree, [randn(1) for _ in 1:5])

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

##
end