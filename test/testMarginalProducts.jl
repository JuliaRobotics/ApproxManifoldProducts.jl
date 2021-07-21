
using ApproxManifoldProducts
using TensorCast
using Test

##

@testset "test product of marginals" begin

## random data

pts1 = [rand(2) .- 10.0 for _ in 1:100]
pts2 = [rand(2) for _ in 1:100]
pts3 = [rand(2) .+ 10.0 for _ in 1:100]

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

pts = getPoints(P)
@cast pGM[i,j] := pts[j][i]


@test 70 < sum(-12 .< pGM[1,:] .< -8)
@test 70 < sum(  8 .< pGM[2,:] .< 12 )

##
end



#