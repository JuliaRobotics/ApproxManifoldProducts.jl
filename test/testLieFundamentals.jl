
using Test
using Manifolds
using ApproxManifoldProducts


##


# [Chirikjian 2012, Vol2, Vol2 p.40], specifically for SO(3)
# @test isapprox(Jl, R*Jr)
# @test isapprox(Jl*inv(Jr), R) == [Ad(R)]
# @test isapprox(Jl, Jr')
# @test det(Jl) == det(Jr) == 2*(1-cos||x||)/(||x||^2)


@testset "SO(3) Vector transport via Jacobians and Lie adjoints" begin 
##

M = SpecialOrthogonal(3)
p = Identity(M)
Xc = [1.,0,0]
X = hat(M, p, Xc)       # random algebra element
d = hat(M, p, [0,0,1.]) # direction in algebra

## piggy back utility tests used in construction of transports

# Ad_vee and ad_vee are matrices which use multiplication to operate Ad and ad respectively
# @test Ad_vee(exp_G(-0.5*coord_u)) == exp(-0.5*ad_vee(coord_u))
@test isapprox(
  ApproxManifoldProducts.AdjointMatrix(M, exp_lie(M,-0.5*X)),
  exp(-0.5*ApproxManifoldProducts.adjointMatrix(M, X))
)


## parallel transport without curvature correction

Y = parallel_transport_direction(M, p, X, d) # compare to Lie ad and Ad and P, bottom [Mahony 2024 p3]
Yc_ = ApproxManifoldProducts.transportMatrix(M, d) * Xc
Y_ = hat(M, p, Yc_)

@test isapprox(Y, Y_)
@test isapprox(
  Y,
  ApproxManifoldProducts.transportAction_matrices_direction(M, d, X)
)


## With curvature correction

# compute transported coordinates (with Mahony 2nd order curvature correction)
# is this also a push-forward
_Y_ = ApproxManifoldProducts.Jl_trivialized(M, d)*Xc
# _Y_ = Jl_trivialized(M, d, Xc)

# Approx check for approx curvature correctin _Y_ vs. transport in direction (wo curvature corr) Y
@test isapprox(_Y_, vee(M, p, Y); atol=1e-1)

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