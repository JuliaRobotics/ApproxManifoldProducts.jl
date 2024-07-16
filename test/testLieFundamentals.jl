
using Test
using Manifolds
using ApproxManifoldProducts
using LinearAlgebra


##


# [Chirikjian 2012, Vol2, Vol2 p.40], specifically for SO(3)
# @test isapprox(Jl, R*Jr)
# @test isapprox(Jl*inv(Jr), R) == [Ad(R)]
# @test isapprox(Jl, Jr')
# @test det(Jl) == det(Jr) == 2*(1-cos||x||)/(||x||^2)


@testset "SO(2) Vector transport (w curvature) via Jacobians and Lie adjoints" begin
##

M = SpecialOrthogonal(2)
p = Identity(M)
Xc = 0.5*randn(1)
X = hat(M, p, Xc)   # random algebra element
d̂ = 0.5*randn(1) 
d = hat(M, p, d̂)    # direction in algebra


ApproxManifoldProducts.ad_lie(M, X)
# @test isapprox(
#   ApproxManifoldProducts.ad_lie(M, X),
#   ApproxManifoldProducts.ad(M, X)
# )

# ptcMat = ApproxManifoldProducts.parallel_transport_curvature_2nd_lie(M, d)

@error "Missing SO(2) vector transport numerical verify tests"
# # # @test isapprox(Jl*inv(Jr), R) == [Ad(R)]
# # Jr = ptcMat
# # Jl = ptcMat'
# # @test isapprox(
# #   Jl*inv(Jr),
# #   ApproxManifoldProducts.Ad(M,exp_lie(M,d));
# #   atol=1e-5
# # )


##
end


@testset "SO(3) Vector transport (w curvature) via Jacobians and Lie adjoints" begin 
##

M = SpecialOrthogonal(3)
p = Identity(M)
Xc = [1.,0,0]
X = hat(M, p, Xc)   # random algebra element
d̂ = [0,0,1.]
d = hat(M, p, d̂)    # direction in algebra

## piggy back utility tests used in construction of transports

@test isapprox(
  ApproxManifoldProducts.ad_lie(M, X),
  ApproxManifoldProducts.ad(M, X)
)

# Ad_vee and ad_vee are matrices which use multiplication to operate Ad and ad respectively
# @test Ad_vee(exp_G(-0.5*coord_u)) == exp(-0.5*ad_vee(coord_u))
@test isapprox(
  ApproxManifoldProducts.Ad(M, exp_lie(M,-0.5*X)),
  exp(-0.5*ApproxManifoldProducts.ad(M, X))
)

## parallel transport without curvature correction

Y = parallel_transport_direction(M, p, X, d) # compare to Lie ad and Ad and P, bottom [Mahony 2024 p3]
ptcMat_ = ApproxManifoldProducts.parallel_transport_direction_lie(M, d)
Yc_ = ptcMat_ * Xc
Y_ = hat(M, p, Yc_)

@test isapprox(
  Y, 
  Y_
)
@test isapprox(
  Y,
  ApproxManifoldProducts.parallel_transport_direction_lie(M, d, X)
)

# @test isapprox(Jl*inv(Jr), R) == [Ad(R)]
Jr = ptcMat_
Jl = ptcMat_'
@test isapprox(
  Jl*inv(Jr),
  ApproxManifoldProducts.Ad(M,exp_lie(M,d));
  atol=1e-8
)

# @test det(Jl) == det(Jr) == 2*(1-cos||x||)/(||x||^2) # from [Chirikjian 2012, Vol 2, ~pg.40] 
@test isapprox(
  det(Jl),
  det(Jr)
)
# @test isapprox(
#   det(Jl),
#   2*(1-cos(norm(d̂)))/(norm(d̂)^2);
#   atol=1e-8
# )

## With curvature correction

# compute transported coordinates (with Mahony 2nd order curvature correction)
# is this also a push-forward
ptcMat = ApproxManifoldProducts.parallel_transport_curvature_2nd_lie(M, d)
_Y_ = ptcMat*Xc
# _Y_ = parallel_transport_curvature_2nd_lie(M, d, Xc)

# Approx check for approx curvature correctin _Y_ vs. transport in direction (wo curvature corr) Y
@test isapprox(_Y_, vee(M, p, Y); atol=1e-1)

# @test isapprox(Jl*inv(Jr), R) == [Ad(R)]
Jr = ptcMat
Jl = ptcMat'
@test isapprox(
  Jl*inv(Jr),
  ApproxManifoldProducts.Ad(M,exp_lie(M,d));
  atol=1e-8
)

# @test det(Jl) == det(Jr) == 2*(1-cos||x||)/(||x||^2) # from [Chirikjian 2012, Vol 2, ~pg.40] 
@test isapprox(
  det(Jl),
  det(Jr)
)
# @test isapprox(
#   det(Jl),
#   2*(1-cos(norm(d̂)))/(norm(d̂)^2);
#   atol=1e-8
# )


# ----- turnkey version -----

@test isapprox(
  _Y_,
  ApproxManifoldProducts.parallel_transport_best(M,p,X,d); 
  atol=1e-8
)



@warn "TODO find Manifolds operation equivalent to Mahony 2024, perhaps `vector_transport_along`?"
# # https://juliamanifolds.github.io/ManifoldsBase.jl/stable/vector_transports/
# # https://juliamanifolds.github.io/ManifoldsBase.jl/stable/vector_transports/#ManifoldsBase.SchildsLadderTransport
# c = [0,0,1.]
# vector_transport_along(M, identity_element(M), X, c)
# vector_transport_direction(M, p, X, d)

# q = exp(M, p, d)
# vector_transport_to(M, p, X, q)

# c = mid_point(M, q, d)

# vector_transport_direction(M, p, X, vee(M,p,c))


##
end


@testset "SE(2) Vector transport (w curvature) via Jacobians and Lie adjoints" begin
##

M = SpecialEuclidean(2)
p = Identity(M)
Xc = 0.5*randn(3)
X = hat(M, p, Xc)   # random algebra element
d̂ = 0.5*randn(3) 
d = hat(M, p, d̂)    # direction in algebra

@test isapprox(
  ApproxManifoldProducts.ad_lie(M, X),
  ApproxManifoldProducts.ad(M, X)
)

ptcMat = ApproxManifoldProducts.parallel_transport_curvature_2nd_lie(M, d)

@error "Missing SE(2) vector transport numerical verify tests"
# # @test isapprox(Jl*inv(Jr), R) == [Ad(R)]
# Jr = ptcMat
# Jl = ptcMat'
# @test isapprox(
#   Jl*inv(Jr),
#   ApproxManifoldProducts.Ad(M,exp_lie(M,d));
#   atol=1e-5
# )


##
end


@testset "SE(3) Vector transport (w curvature) via Jacobians and Lie adjoints" begin
##


M = SpecialEuclidean(3)
p = Identity(M)
Xc = 0.5*randn(6)
X = hat(M, p, Xc)   # random algebra element
d̂ = 0.5*randn(6) 
d = hat(M, p, d̂)    # direction in algebra

@test isapprox(
  ApproxManifoldProducts.ad_lie(M, X),
  ApproxManifoldProducts.ad(M, X)
)

ptcMat = ApproxManifoldProducts.parallel_transport_curvature_2nd_lie(M, d)

@error "Missing SE(3) vector transport numerical verify tests"
# # @test isapprox(Jl*inv(Jr), R) == [Ad(R)]
# Jr = ptcMat
# Jl = ptcMat'
# @test isapprox(
#   Jl*inv(Jr),
#   ApproxManifoldProducts.Ad(M,exp_lie(M,d));
#   atol=1e-5
# )


##
end

#