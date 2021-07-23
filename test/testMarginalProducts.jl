
using Manifolds
using ApproxManifoldProducts
using TensorCast
using Test

##

@testset "test product of marginals" begin

## random data

pts1 = [randn(2) .- 10.0 for _ in 1:100]
pts2 = [randn(2) for _ in 1:100]
pts3 = [randn(2) .+ 10.0 for _ in 1:100]

## get different marginals

M = Euclidean(2)

P1 = marginal(manikde!(pts1, M), [1;])
P2 =          manikde!(pts2, M)
P3 = marginal(manikde!(pts3, M), [2;])

##

@test_broken false
# P = manifoldProduct([P1;P2;P3])
@error "manifoldProduct needs complete point type for keyword `oldPoints`, current tests assume no marginal beliefs at front of product array"
P = manifoldProduct([P2;P1;P3])


## check the results

pts = getPoints(P)
@cast pGM[i,j] := pts[j][i]

@test 80 < sum(-10 .< pGM[1,:] .< 0)
@test 80 < sum(0 .< pGM[2,:] .< 10 )


## check product of only partials/marginals

P_ = manifoldProduct([P1;P3])

pts = getPoints(P_)
@cast pGM[i,j] := pts[j][i]


@test 70 < sum(-12 .< pGM[1,:] .< -8)
@test 70 < sum(  8 .< pGM[2,:] .< 12 )

## 

N = 5
M = TranslationGroup(3)

pts4 = [3*randn(3) .- 10.0 for _ in 1:N]
(x->x[2:3].=-100.0).(pts4)
pts5 = [3*randn(3) .+ 10.0 for _ in 1:N]
(x->x[1:2].=100.0).(pts5)

P4 = marginal(manikde!(pts4, M), [1;])
P5 = marginal(manikde!(pts5, M), [3;])


# test duplication
pts4_ = [3*randn(3) .- 10.0 for _ in 1:N]
(x->x[2:3].=-100.0).(pts4_)
pts5_ = [3*randn(3) .+ 10.0 for _ in 1:N]
(x->x[1:2].=100.0).(pts5_)

P4_ = marginal(manikde!(pts4_, M), [1;])
P5_ = marginal(manikde!(pts5_, M), [3;])

##

P45 = manifoldProduct([P4;P4_;P5;P5_])


##
end



#