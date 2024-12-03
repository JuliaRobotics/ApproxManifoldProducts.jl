
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
  @show testval = isapprox(0, _ax_ERR; atol = 8/length(ax_CC))
  @assert testval "Spot check failed on eigen split of manifold points, the estimated point rotation matrix did not match construction. length(ax_CC)=$(length(ax_CC))"

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

@testset "Test evaluate MvNormalKernel" begin
##

M = TranslationGroup(1)
ker = AMP.MvNormalKernel([0.0], [0.5;;])
@test isapprox(
  AMP.evaluate(M, ker, [0.1]),
  pdf(MvNormal(mean(ker), cov(ker)), [0.1])
)


# Test wrapped cicular distribution 
function pdf_wrapped_normal(μ, σ, θ; nwrap=1000) 
  s = 0.0
  for k = -nwrap:nwrap
    s += exp(-(θ - μ + 2pi*k)^2 / (2*σ^2))
  end
  return 1/(σ*sqrt(2pi)) * s
end 

M = RealCircleGroup()
ker = AMP.MvNormalKernel([0.0], [0.1;;])
@test isapprox(
  AMP.evaluate(M, ker, [0.1]),
  pdf_wrapped_normal(mean(ker)[], sqrt(cov(ker))[], 0.1)
)

ker = AMP.MvNormalKernel([0], [2.0;;])
@test isapprox(
  AMP.evaluate(M, ker, [0.]),
  AMP.evaluate(M, ker, [2pi])
)
#TODO wrapped normal distributions broken
@test_broken isapprox(
  pdf_wrapped_normal(mean(ker)[], sqrt(cov(ker))[], pi),
  AMP.evaluate(M, ker, [pi])
)
@test_broken isapprox(
  pdf_wrapped_normal(mean(ker)[], sqrt(cov(ker))[], 0),
  AMP.evaluate(M, ker, [0.])
)

##
M = SpecialEuclidean(2; vectors=HybridTangentRepresentation())
ε = identity_element(M)
Xc = [10, 20, 0.1]
p = exp(M, ε, hat(M, ε, Xc))
kercov = diagm([0.5, 2.0, 0.1].^2)
ker = AMP.MvNormalKernel(p, kercov)
@test isapprox(
  AMP.evaluate(M, ker, p),
  pdf(MvNormal(Xc, cov(ker)), Xc)
)

Xc = [10, 22, -0.1]
q = exp(M, ε, hat(M, ε, Xc))

@test isapprox(
  pdf(MvNormal(cov(ker)), [0,0,0]),
  AMP.evaluate(M, ker, p)
)

X = log(M, ε, Manifolds.compose(M, inv(M, p), q))
Xc_e = vee(M, ε, X)
pdf_local_coords = pdf(MvNormal(cov(ker)), Xc_e)

@test isapprox(
  pdf_local_coords,
  AMP.evaluate(M, ker, q),
)

delta_c = AMP.distanceMalahanobisCoordinates(M, ker, q)
X = log(M, ε, Manifolds.compose(M, inv(M, p), q))
Xc_e = vee(M, ε, X)
malad_t = Xc_e'*inv(kercov)*Xc_e
# delta_t = [10, 20, 0.1] - [10, 22, -0.1] 
@test isapprox(
  malad_t,
  delta_c'*delta_c;
  atol=1e-10
)

malad2 = AMP.distanceMalahanobisSq(M,ker,q)
@test isapprox(
  malad_t,
  malad2;
  atol=1e-10
)

rbfd = AMP.ker(M, ker, q, 0.5, AMP.distanceMalahanobisSq)
@test isapprox(
  exp(-0.5*malad_t),
  rbfd;
  atol=1e-10
)

# NOTE 'global' distribution would have been 
X = log(M, mean(ker), q) 
Xc_e = vee(M, ε, X)
pdf_global_coords = pdf(MvNormal(cov(ker)), Xc_e)


##
end

@testset "Basic ManellicTree manifolds construction and evaluations" begin
## 


M = TranslationGroup(1)
ε = identity_element(M)
dis = MvNormal([3.0], diagm([1.0].^2)) 
Cpts = [rand(dis) for _ in 1:128]
pts = map(c->exp(M, ε, hat(M, ε, c)), Cpts)
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw = [0.2;;],  kernel=AMP.MvNormalKernel)

##
p = exp(M, ε, hat(M, ε, [3.0]))
y_amp = AMP.evaluate(mtree, p)

y_pdf = pdf(dis, [3.0])

@test isapprox(y_amp, y_pdf; atol=0.1)

# ps = [[p] for p = -0:0.01:6]
# ys_amp = map(p->AMP.evaluate(mtree, exp(M, ε, hat(M, ε, p))), ps)
# ys_pdf = pdf(dis, ps)

# lines(first.(ps), ys_pdf)
# lines!(first.(ps), ys_amp)

# lines!(first.(ps), ys_pdf)
# lines(first.(ps), ys_amp)
##

M = SpecialOrthogonal(2)
ε = identity_element(M)
dis = MvNormal([0.0], diagm([0.1].^2)) 
Cpts = [rand(dis) for _ in 1:128]
pts = map(c->exp(M, ε, hat(M, ε, c)), Cpts)
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw = [0.005;;], kernel=AMP.MvNormalKernel)

##
p = exp(M, ε, hat(M, ε, [0.1]))
y_amp = AMP.evaluate(mtree, p)

y_pdf = pdf(dis, [0.1])

@test isapprox(y_amp, y_pdf; atol=0.5)

ps = [[p] for p = -0.3:0.01:0.3]
ys_amp = map(p->AMP.evaluate(mtree, exp(M, ε, hat(M, ε, p))), ps)
ys_pdf = pdf(dis, ps)

# lines(first.(ps), ys_pdf)
# lines!(first.(ps), ys_amp)


M = SpecialEuclidean(2; vectors=HybridTangentRepresentation())
ε = identity_element(M)
dis = MvNormal([10,20,0.1], diagm([0.5,2.0,0.1].^2)) 
Cpts = [rand(dis) for _ in 1:128]
pts = map(c->exp(M, ε, hat(M, ε, c)), Cpts)
mtree = ApproxManifoldProducts.buildTree_Manellic!(M, pts; kernel_bw = diagm([0.05,0.2,0.01]), kernel=AMP.MvNormalKernel)

##
p = exp(M, ε, hat(M, ε, [10, 20, 0.1]))
y_amp = AMP.evaluate(mtree, p)
y_pdf = pdf(dis, [10,20,0.1])
# check kde eval is within 20% of true value
y_err = y_amp - y_pdf
@show y_pdf
if !isapprox(0, y_err; atol=0.2*y_pdf)
  @warn "soft test failure for approx function vs. true Normal density function evaluation"
  @test_broken false
end

##
end


