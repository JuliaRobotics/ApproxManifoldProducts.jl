

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
@testset "Test Product the brute force way, SpecialEuclidean(2; vectors=HybridTangentRepresentation())" begin

M = SpecialEuclidean(2; vectors=HybridTangentRepresentation())
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


@testset "Rotated covariance product major axis checks, SpecialEuclidean(2; vectors=HybridTangentRepresentation())" begin
##

M = SpecialEuclidean(2; vectors=HybridTangentRepresentation())
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
@show mean(kerpq)

@test isapprox(
  pi/6,
  Rot_.RotMatrix(SMatrix{2,2,Float64}(mean(kerpq).x[2])) |> Rot_.rotation_angle;
  atol=1e-8
)


##
end


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





#
