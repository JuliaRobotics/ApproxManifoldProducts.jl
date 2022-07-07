# test manifold conventions


using Test
using Manifolds

import Rotations as _Rot

##
@testset "test Manifolds.jl conventions" begin
##

M = SpecialEuclidean(2)
e0 = identity_element(M)

# body to body next
bTb_ = ArrayPartition([10.0;0], _Rot.RotMatrix(pi/2))

# drive in a clockwise square from the origin
wTb0 = ArrayPartition([100.0;0], _Rot.RotMatrix(0.0))
wTb1 = Manifolds.compose(M, wTb0, bTb_)    # right
wTb2 = Manifolds.compose(M, wTb1, bTb_)    # top right
wTb3 = Manifolds.compose(M, wTb2, bTb_)    # top
wTb4 = Manifolds.compose(M, wTb3, bTb_)    # origin

wCb0 = vee(M, e0, log(M, e0, wTb0))
wCb1 = vee(M, e0, log(M, e0, wTb1))
wCb2 = vee(M, e0, log(M, e0, wTb2))
wCb3 = vee(M, e0, log(M, e0, wTb3))
wCb4 = vee(M, e0, log(M, e0, wTb4))

##

# check the favorable result
@test isapprox( [100,0.,0],    wCb0 ; atol=1e-6 )
@test isapprox( [110,0.,pi/2], wCb1 ; atol=1e-6 )
@test isapprox( [110.,10,pi],  wCb2 ; atol=1e-6 ) || isapprox( [110,10,-pi], wCb2 ; atol=1e-6 )
@test isapprox( [100.,10,-pi/2], wCb3 ; atol=1e-6 )
@test isapprox( [100,0,0.], wCb4 ; atol=1e-6 )


## check that the inverse breaks


# Use opposite convention from above to show it is wrong
wTb0 = ArrayPartition([100.0;0], _Rot.RotMatrix(0.0))
wTb1 = Manifolds.compose(M, bTb_, wTb0)    # right
wTb2 = Manifolds.compose(M, bTb_, wTb1)    # top right
wTb3 = Manifolds.compose(M, bTb_, wTb2)    # top
wTb4 = Manifolds.compose(M, bTb_, wTb3)    # origin

wCb0 = vee(M, e0, log(M, e0, wTb0))
wCb1 = vee(M, e0, log(M, e0, wTb1))
wCb2 = vee(M, e0, log(M, e0, wTb2))
wCb3 = vee(M, e0, log(M, e0, wTb3))
wCb4 = vee(M, e0, log(M, e0, wTb4))

# check the negative result
@test  isapprox( [100,0.,0],      wCb0 ; atol=1e-6 )
@test !isapprox( [110,0.,pi/2],   wCb1 ; atol=1e-6 )
@test !(isapprox( [110.,10,pi],    wCb2 ; atol=1e-6 ) || isapprox( [110,10,-pi], wCb2 ; atol=1e-6 ))
@test !isapprox( [100.,10,-pi/2], wCb3 ; atol=1e-6 )
@test  isapprox( [100,0,0.],      wCb4 ; atol=1e-6 )


##
end