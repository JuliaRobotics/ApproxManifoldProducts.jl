# test manifold conventions


using Test
import Manifolds as MF
import LieGroups as LGr
import Rotations as _Rot

##
@testset "test MF.jl conventions" begin
##

M = MF.SpecialEuclidean(2; vectors = MF.HybridTangentRepresentation())
e0 = MF.identity_element(M)

# body to body next
bTb_ = ArrayPartition([10.0;0], _Rot.RotMatrix(pi/2))

# drive in a clockwise square from the origin
wTb0 = ArrayPartition([100.0;0], _Rot.RotMatrix(0.0))
wTb1 = MF.compose(M, wTb0, bTb_)    # right
wTb2 = MF.compose(M, wTb1, bTb_)    # top right
wTb3 = MF.compose(M, wTb2, bTb_)    # top
wTb4 = MF.compose(M, wTb3, bTb_)    # origin

wCb0 = MF.vee(M, e0, MF.log(M, e0, wTb0))
wCb1 = MF.vee(M, e0, MF.log(M, e0, wTb1))
wCb2 = MF.vee(M, e0, MF.log(M, e0, wTb2))
wCb3 = MF.vee(M, e0, MF.log(M, e0, wTb3))
wCb4 = MF.vee(M, e0, MF.log(M, e0, wTb4))

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
wTb1 = MF.compose(M, bTb_, wTb0)    # right
wTb2 = MF.compose(M, bTb_, wTb1)    # top right
wTb3 = MF.compose(M, bTb_, wTb2)    # top
wTb4 = MF.compose(M, bTb_, wTb3)    # origin

wCb0 = MF.vee(M, e0, MF.log(M, e0, wTb0))
wCb1 = MF.vee(M, e0, MF.log(M, e0, wTb1))
wCb2 = MF.vee(M, e0, MF.log(M, e0, wTb2))
wCb3 = MF.vee(M, e0, MF.log(M, e0, wTb3))
wCb4 = MF.vee(M, e0, MF.log(M, e0, wTb4))

# check the negative result
@test  isapprox( [100,0.,0],      wCb0 ; atol=1e-6 )
@test !isapprox( [110,0.,pi/2],   wCb1 ; atol=1e-6 )
@test !(isapprox( [110.,10,pi],    wCb2 ; atol=1e-6 ) || isapprox( [110,10,-pi], wCb2 ; atol=1e-6 ))
@test !isapprox( [100.,10,-pi/2], wCb3 ; atol=1e-6 )
@test  isapprox( [100,0,0.],      wCb4 ; atol=1e-6 )


##
end