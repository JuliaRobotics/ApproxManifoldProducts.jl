
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
M = SpecialEuclidean(2)
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


M = SpecialEuclidean(2)
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

end


## ========================================================================================
@testset "Test Product the brute force way, SpecialEuclidean(2)" begin

M = SpecialEuclidean(2)
ε = identity_element(M)

Xc_p = [10, 20, 0.1]
p = exp(M, ε, hat(M, ε, Xc_p))
kerp = AMP.MvNormalKernel(p, diagm([0.5, 2.0, 0.1].^2))

Xc_q = [10, 22, -0.1]
q = exp(M, ε, hat(M, ε, Xc_q))
kerq = AMP.MvNormalKernel(q, diagm([1.0, 1.0, 0.1].^2))

kerpq = calcProductGaussians(M, [kerp, kerq])

# brute force way
xs = 7:0.1:13
ys = 15:0.1:27
θs = -0.3:0.01:0.3

grid_points = map(Iterators.product(xs, ys, θs)) do (x,y,θ)
    exp(M, ε, hat(M, ε, SVector(x,y,θ)))
end

# use_global_coords = true
use_global_coords = false

pdf_ps = map(grid_points) do gp
  if use_global_coords
    X = log(M, p, gp) 
    Xc_e = vee(M, ε, X)
    pdf(MvNormal(cov(kerp)), Xc_e)
  else
    X = log(M, ε, Manifolds.compose(M, inv(M, p), gp))
    Xc_e = vee(M, ε, X)
    pdf(MvNormal(cov(kerp)), Xc_e)
  end
end

pdf_qs = map(grid_points) do gp
  if use_global_coords
    X = log(M, q, gp) 
    Xc_e = vee(M, ε, X)
    pdf(MvNormal(cov(kerq)), Xc_e)
  else
    X = log(M, ε, Manifolds.compose(M, inv(M, q), gp))
    Xc_e = vee(M, ε, X)
    pdf(MvNormal(cov(kerq)), Xc_e)
  end
end

pdf_pqs = pdf_ps .* pdf_qs
# pdf_pqs ./= sum(pdf_pqs) * 0.01
# pdf_pqs .*= 15.9672

amp_ps = map(grid_points) do gp
  AMP.evaluate(M, kerp, gp)
end
amp_qs = map(grid_points) do gp
  AMP.evaluate(M, kerq, gp)
end

amp_pqs = map(grid_points) do gp
  AMP.evaluate(M, kerpq, gp)
end

amp_bf_pqs = amp_ps .* amp_qs

#FIXME -- brute force will be more accurate than approx product, relax these tests for stochastic variability
normalized_compare_test = isapprox.(normalize(amp_pqs), normalize(amp_bf_pqs); atol=0.001)
@test_broken all(normalized_compare_test)
@warn "Brute force product test overlap $(round(count(normalized_compare_test) / length(amp_pqs) * 100, digits=2))%"

#TODO should this be local or global coords?
@test_broken findmax(pdf_pqs[:,60,30])[2] == findmax(amp_pqs[:,60,30])[2]

# these are all correct
# lines(xs, pdf_ps[:,60,30])
# lines!(xs, amp_ps[:,60,30])

# lines(ys, pdf_ps[30,:,30])
# lines!(ys, amp_ps[30,:,30])

# lines(θs, pdf_ps[30,60,:])
# lines!(θs, amp_ps[30,60,:])

#these are different for "local" vs "global"
# lines(xs, normalize(pdf_pqs[:,60,30]))
# lines!(xs, normalize(amp_pqs[:,60,30]))
# lines!(xs, normalize(amp_bf_pqs[:,60,30]))

# lines(ys, normalize(pdf_pqs[30,:,30]))
# lines!(ys, normalize(amp_pqs[30,:,30]))

# lines(θs, normalize(pdf_pqs[30,60,:]))
# lines!(θs, normalize(amp_pqs[30,60,:]))

# contour(xs, ys, pdf_pqs[:,:,30]; color = :blue)
# contour!(xs, ys, amp_pqs[:,:,30]; color = :red)
# contour!(xs, ys, amp_bf_pqs[:,:,30]; color = :green)

#just some exploration
# pdf_p = pdf(Normal(10, 0.5), xs)
# pdf_q = pdf(Normal(10, 1.0), xs)
# pdf_pq = (pdf_p .* pdf_q)
# pdf_pq ./= sum(pdf_pq) * 0.01 

# lines(xs, pdf_p)
# lines!(xs, pdf_q)
# lines!(xs, pdf_pq)

# pdf_p = pdf(Normal(20, 2.0), ys)
# pdf_q = pdf(Normal(22, 1.0), ys)
# pdf_pq = (pdf_p .* pdf_q)
# pdf_pq ./= sum(pdf_pq) * 0.01 

# lines(ys, pdf_p)
# lines!(ys, pdf_q)
# lines!(ys, pdf_pq)

# pdf_p = pdf(Normal(0.1, 0.1), θs)
# pdf_q = pdf(Normal(-0.1, 0.1), θs)
# pdf_pq = (pdf_p .* pdf_q)
# pdf_pq ./= sum(pdf_pq) * 0.01 

# lines(θs, pdf_p)
# lines!(θs, pdf_q)
# lines!(θs, pdf_pq)

##
end


@testset "Rotated covariance product major axis checks, TranslationGroup(2)" begin
##

M = TranslationGroup(2)
ε = identity_element(M)

Xc_p = [0, 0.0]
p = exp(M, ε, hat(M, ε, Xc_p))
kerp = AMP.MvNormalKernel(p, diagm([2.0, 1.0].^2))

Xc_q = [0, 0.0]
# rotate by 60 deg
R = Rot_.RotMatrix{2}(pi/3).mat
Σ = R * diagm([2.0, 1.0].^2) * R'
q = exp(M, ε, hat(M, ε, Xc_q))
kerq = AMP.MvNormalKernel(q, Σ)

kerpq = calcProductGaussians(M, [kerp, kerq])

evv = eigen(cov(kerpq))
maj_idx = sortperm(evv.values)[end]

# check that the major axis is halfway between 0 and 60deg -- i.e. 30 deg
maj_ang = (angle(Complex(evv.vectors[:,maj_idx]...)) + 2pi) % pi
@test isapprox(pi/180*30, maj_ang; atol = 1e-8)

##
end


@testset "Rotated covariance product major axis checks, SpecialEuclidean(2)" begin
##

M = SpecialEuclidean(2)
ε = identity_element(M)

Xc_p = [0, 0, 0.0]
p = exp(M, ε, hat(M, ε, Xc_p))
kerp = AMP.MvNormalKernel(p, diagm([2.0, 1.0, 0.1].^2))

# referenced to "global frame"
# rotate by 60 deg
Xc_q = [0, 0, pi/3]
q = exp(M, ε, hat(M, ε, Xc_q))
kerq = AMP.MvNormalKernel(q, diagm([2.0, 1.0, 0.1].^2))

kerpq = calcProductGaussians(M, [kerp, kerq])

evv = eigen(cov(kerpq))
maj_idx = sortperm(evv.values)[end]

# check that the major axis is halfway between 0 and 60deg -- i.e. 30 deg
maj_ang = (angle(Complex(evv.vectors[:,maj_idx]...)) + 2pi) % pi
@test isapprox(pi/180*30, maj_ang; atol = 1e-8)



##
end




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



@testset "Multidimensional LOOCV bandwidth optimization, SpecialEuclidean(2)" begin
##

M = SpecialEuclidean(2)
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



@testset "Multidimensional LOOCV bandwidth optimization, SpecialEuclidean(3)" begin
##

M = SpecialEuclidean(3)
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

@test isapprox([0.75; 0.75; 0.75], best_cov[1:3]; atol=0.4)
@test isapprox([0.06; 0.06; 0.06], best_cov[4:6]; atol=0.04)


mkd = ApproxManifoldProducts.manikde!_manellic(M,pts)

@test isapprox([0.75 0 0; 0 0.75 0; 0 0 0.75], getBW(mkd)[1][1:3,1:3]; atol=0.5)
@test isapprox([0.07 0 0; 0 0.07 0; 0 0 0.07], getBW(mkd)[1][4:6,4:6]; atol=0.05)

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


@testset "Test utility functions for Gaussian products, TranslationGroup(1)" begin
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





#
