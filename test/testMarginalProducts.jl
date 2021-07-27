
using Manifolds
using ApproxManifoldProducts
using TensorCast
using LinearAlgebra
using Test

##

@testset "test calcProductGaussians" begin
##

M = TranslationGroup(2)
u = [[1;1.0], [0.0;0]]
c = [([1.0;1]), ([1.0;1])]

u_, C_ = calcProductGaussians(M, u, c)
@test isapprox( u_, [0.5, 0.5])
@test isapprox( C_, [0.5 0.0; 0.0 0.5])

##
end


@testset "comparison test with basic product" begin
## simply multiply two beliefs, sim2

N = 10
d = 2
M = TranslationGroup(d)

#densities to multiply
pts1 = [randn(d) for _ in 1:N]
P1 = manikde!(M, pts1, bw=[1;1.0])

pts2 = [randn(d)  for _ in 1:N]
P2 = manikde!(M, pts2, bw=[1;1.0])

##

sl = Vector{Vector{Int}}()

P12 = manifoldProduct([P1;P2], recordLabels=true, selectedLabels=sl, addEntropy=false)

# @test isapprox( mean(P12)[1], 0, atol=1 )
# @test isapprox( mean(P12)[2], 0, atol=1 )

(x->println()).(1:5)
@show sl;

P12


## validate selected labels are working properly, with addEntropy=false

for sidx in 1:N

  bw1 = getBW(P1)[:,1] .^ 2
  bw2 = getBW(P2)[:,1] .^ 2

  u1 = pts1[sl[sidx][1]]
  u2 = pts2[sl[sidx][2]]

  u12, = calcProductGaussians(M, [u1,u2], [bw1,bw2]);

  @test isapprox( u12, getPoints(P12)[sidx] )

end

##
end


@testset "test dim=2 product with one partial/marginal" begin
## basic test one full with one partial

N = 10
d = 2
M = TranslationGroup(d)

#densities to multiply
pts1 = [randn(d) for _ in 1:N]
P1 = manikde!(M, pts1)

pts2 = [randn(d) for _ in 1:N]
(x->(x[2]+=100)).(pts2)
P2_ = manikde!(M, pts2, partial=[1;])

##

sl = Vector{Vector{Int}}()

P12_ = manifoldProduct([P1;P2_], recordLabels=true, selectedLabels=sl, addEntropy=false)

@test isapprox( mean(P12_)[1], 0, atol=1 )
@test isapprox( mean(P12_)[2], 0, atol=1 )

(x->println()).(1:5)
@show sl

P12_

##

for sidx in 1:N

  bw1 = getBW(P1)[:,1] .^2
  bw2 = getBW(P2_)[:,1] .^2

  u1 = pts1[sl[sidx][1]]
  u2 = pts2[sl[sidx][2]]

  u12, = calcProductGaussians(M, [u1,u2], [bw1,bw2]);

  @test isapprox( u12[1], getPoints(P12_)[sidx][1] , atol=0.1)
  @test isapprox( pts1[sl[sidx][1]][2], getPoints(P12_)[sidx][2] )

end

##
end


@testset "test dim=2 product with one full and two similar marginals" begin
## basic test one full with one partial

N = 5
d = 2
M = TranslationGroup(d)

#densities to multiply
pts1 = [randn(d) for _ in 1:N]
P1 = manikde!(M, pts1)

pts2 = [randn(d) for _ in 1:N]
(x->(x[2]+=100)).(pts2)
P2_ = manikde!(M, pts2, partial=[1;])

pts3 = [randn(d) for _ in 1:N]
(x->(x[2]+=100)).(pts3)
P3_ = manikde!(M, pts3, partial=[1;])


##

sl = Vector{Vector{Int}}()

P123_ = manifoldProduct([P1;P2_;P3_], recordLabels=true, selectedLabels=sl, addEntropy=false)

@test isapprox( mean(P123_)[1], 0, atol=1 )
@test isapprox( mean(P123_)[2], 0, atol=1 )

(x->println()).(1:5)
@show sl

P123_

##

for sidx in 1:N

  bw1 = getBW(P1)[:,1] .^2
  bw2 = getBW(P2_)[:,1] .^2
  bw3 = getBW(P3_)[:,1] .^2

  u1 = pts1[sl[sidx][1]]
  u2 = pts2[sl[sidx][2]]
  u3 = pts3[sl[sidx][3]]

  u123, = calcProductGaussians(M, [u1,u2,u3], [bw1,bw2,bw3]);

  @test isapprox( u123[1], getPoints(P123_)[sidx][1] , atol=0.1)
  @test isapprox( pts1[sl[sidx][1]][2], getPoints(P123_)[sidx][2] )

end

##
end


@testset "test dim=2 product with one full and two different marginals" begin
## random data

N = 10
M = TranslationGroup(2)

pts1 = [randn(2) .- 10.0 for _ in 1:N]
pts2 = [randn(2) for _ in 1:N]
pts3 = [randn(2) .+ 10.0 for _ in 1:N]

# get different marginals

P1 = marginal(manikde!(pts1, M), [1;])
P2 =          manikde!(pts2, M)
P3 = marginal(manikde!(pts3, M), [2;])

##

sl = Vector{Vector{Int}}()
P = manifoldProduct([P2;P1;P3], recordLabels=true, selectedLabels=sl, addEntropy=false)

(x->println()).(1:5)
@show sl;
P

## check the results

pts = getPoints(P)
@cast pGM[i,j] := pts[j][i]

@test 0.8*N < sum(-10 .< pGM[1,:] .< 0)
@test 0.8*N < sum(0 .< pGM[2,:] .< 10 )


## check the selection of labels and resulting Gaussian products are correct

for sidx in 1:N

  bw1 = getBW(P1)[:,1] .^2
  bw2 = getBW(P2)[:,1] .^2
  bw3 = getBW(P3)[:,1] .^2

  # full density first
  u2 = pts2[sl[sidx][1]]
  u1 = pts1[sl[sidx][2]]
  u3 = pts3[sl[sidx][3]]

  u12, = calcProductGaussians(M, [u1,u2], [bw1,bw2]);
  u23, = calcProductGaussians(M, [u2,u3], [bw2,bw3]);

  @test isapprox( u12[1], getPoints(P)[sidx][1])
  @test isapprox( u23[2], getPoints(P)[sidx][2])

end

##
end


@testset "product of only one marginal per each of two dimensions" begin
## random data

N = 10
M = TranslationGroup(2)

pts1 = [randn(2) .- 10.0 for _ in 1:N]
pts3 = [randn(2) .+ 10.0 for _ in 1:N]

# different marginals

P1 = marginal(manikde!(pts1, M), [1;])
P3 = marginal(manikde!(pts3, M), [2;])

## 

sl = Vector{Vector{Int}}()

P_ = manifoldProduct([P1;P3], recordLabels=true, selectedLabels=sl, addEntropy=false)

(x->println()).(1:5)
@show sl

##

pts = getPoints(P_)
@cast pGM[i,j] := pts[j][i]

@test 0.7*N < sum(-12 .< pGM[1,:] .< -8)
@test 0.7*N < sum(  8 .< pGM[2,:] .< 12 )

## check the selection of labels and resulting Gaussian products are correct

for sidx in 1:N

  bw1 = getBW(P1)[:,1] .^2
  bw3 = getBW(P3)[:,1] .^2

  u1 = pts1[sl[sidx][1]]
  u3 = pts3[sl[sidx][2]]

  @test isapprox( u1[1], pts[sidx][1])
  @test isapprox( u3[2], pts[sidx][2])

end

## 
end


@testset "test dim=2 product of only marginals, two per dimension, 4 factors total" begin
##

N = 5
d = 2
M = TranslationGroup(d)

pts4 = [randn(d) .- 10.0 for _ in 1:N]
(x->x[2]=-100.0).(pts4)
pts5 = [randn(d) .+ 10.0 for _ in 1:N]
(x->x[1]=100.0).(pts5)

P4 = marginal(manikde!(pts4, M), [1;])
P5 = marginal(manikde!(pts5, M), [d;])

# test duplication
pts4_ = [randn(d) .- 10.0 for _ in 1:N]
(x->x[2]=-100.0).(pts4_)
pts5_ = [randn(d) .+ 10.0 for _ in 1:N]
(x->x[1]=100.0).(pts5_)

P4_ = marginal(manikde!(pts4_, M), [1;])
P5_ = marginal(manikde!(pts5_, M), [d;])

##

sl = Vector{Vector{Int}}()

P45__ = manifoldProduct([P4;P4_;P5;P5_], recordLabels=true, selectedLabels=sl, addEntropy=false)

(x->println()).(1:5)
@show sl;

P45__

## check the selection of labels and resulting Gaussian products are correct

println("getPoints(P45__) = ")
getPoints(P45__) .|> println
println()

for sidx in 1:N

  bw1 = getBW(P4)[:,1] .^2
  bw2 = getBW(P4_)[:,1] .^2
  bw3 = getBW(P5)[:,1] .^2
  bw4 = getBW(P5_)[:,1] .^2

  u1 = pts4[ sl[sidx][1]]
  u2 = pts4_[sl[sidx][2]]
  u3 = pts5[ sl[sidx][3]]
  u4 = pts5_[sl[sidx][4]]

  u12, = calcProductGaussians(M, [u1,u2], [bw1,bw2]);
  u34, = calcProductGaussians(M, [u3,u4], [bw3,bw4]);

  @show u12[1], u34[2]
  @test isapprox( u12[1], getPoints(P45__)[sidx][1])
  @test isapprox( u34[2], getPoints(P45__)[sidx][2])

end

##
end


@testset "test dim=3 product with one full and two different marginals" begin
## random data

d = 3
N = 10
M = TranslationGroup(3)

pts1 = [randn(d) .- 10.0 for _ in 1:N]
pts2 = [randn(d) for _ in 1:N]
pts3 = [randn(d) .+ 10.0 for _ in 1:N]

## get different marginals

P1 = marginal(manikde!(pts1, M), [1;])
P2 =          manikde!(pts2, M)
P3 = marginal(manikde!(pts3, M), [d;])

##

# @test_broken false
# # P = manifoldProduct([P1;P2;P3])
# @error "manifoldProduct needs complete point type for keyword `oldPoints`, current tests assume no marginal beliefs at front of product array"
sl = Vector{Vector{Int}}()
P = manifoldProduct([P2;P1;P3], recordLabels=true, selectedLabels=sl, addEntropy=false)

(x->println()).(1:5)
@show sl;
P

## check the results

pts = getPoints(P)
@cast pGM[i,j] := pts[j][i]

@test 0.6*N < sum(-10 .< pGM[1,:] .< 0)
@test 0.6*N < sum(  0 .< pGM[3,:] .< 10 )


## check the selection of labels and resulting Gaussian products are correct

for sidx in 1:N

  bw1 = getBW(P1)[:,1] .^2
  bw2 = getBW(P2)[:,1] .^2
  bw3 = getBW(P3)[:,1] .^2

  # full density first
  u2 = pts2[sl[sidx][1]]
  u1 = pts1[sl[sidx][2]]
  u3 = pts3[sl[sidx][3]]

  u12, = calcProductGaussians(M, [u1,u2], [bw1,bw2]);
  u23, = calcProductGaussians(M, [u2,u3], [bw2,bw3]);

  @test isapprox( u12[1], getPoints(P)[sidx][1])
  @test isapprox(  u2[2], getPoints(P)[sidx][2] )
  @test isapprox( u23[3], getPoints(P)[sidx][3])

end

##
end


@testset "test dim=3 product with one full and two different marginals (marginal first in product)" begin
## random data

d = 3
N = 10
M = TranslationGroup(3)

pts1 = [randn(d) .- 10.0 for _ in 1:N]
pts2 = [randn(d) for _ in 1:N]
pts3 = [randn(d) .+ 10.0 for _ in 1:N]

## get different marginals

P1 = marginal(manikde!(pts1, M), [1;])
P2 =          manikde!(pts2, M)
P3 = marginal(manikde!(pts3, M), [d;])

##

sl = Vector{Vector{Int}}()
P = manifoldProduct([P1;P2;P3], recordLabels=true, selectedLabels=sl, addEntropy=false)

(x->println()).(1:5)
@show sl;
P

## check the results

pts = getPoints(P)
@cast pGM[i,j] := pts[j][i]

@test 0.8*N < sum(-10 .< pGM[1,:] .<  0 )
@test 0.8*N < sum(  0 .< pGM[3,:] .< 10 )

## check the selection of labels and resulting Gaussian products are correct

for sidx in 1:N

  bw1 = getBW(P1)[:,1] .^2
  bw2 = getBW(P2)[:,1] .^2
  bw3 = getBW(P3)[:,1] .^2

  # full density first
  u1 = pts1[sl[sidx][1]]
  u2 = pts2[sl[sidx][2]]
  u3 = pts3[sl[sidx][3]]

  u12, = calcProductGaussians(M, [u1,u2], [bw1,bw2]);
  u23, = calcProductGaussians(M, [u2,u3], [bw2,bw3]);

  @test isapprox( u12[1], getPoints(P)[sidx][1])
  @test isapprox(  u2[2], getPoints(P)[sidx][2])
  @test isapprox( u23[3], getPoints(P)[sidx][3])

end

##
end


@testset "test dim=3 product with two different marginals and one open dimension" begin
## random data

d = 3
N = 10
M = TranslationGroup(3)

pts1 = [randn(d) .- 10.0 for _ in 1:N]
pts3 = [randn(d) .+ 10.0 for _ in 1:N]

## get different marginals

P1 = marginal(manikde!(pts1, M), [1;])
P3 = marginal(manikde!(pts3, M), [d;])

##

sl = Vector{Vector{Int}}()
P = manifoldProduct([P1;P3], recordLabels=true, selectedLabels=sl, addEntropy=false)

(x->println()).(1:5)
@show sl;
P

## check the results

pts = getPoints(P)
@cast pGM[i,j] := pts[j][i]

@test 0.7*N < sum(-13 .< pGM[1,:] .< -7 )
@test 0.7*N < sum(  7 .< pGM[3,:] .< 13 )

## check the selection of labels and resulting Gaussian products are correct

for sidx in 1:N

  bw1 = getBW(P1)[:,1] .^2
  bw3 = getBW(P3)[:,1] .^2

  # full density first
  u1 = pts1[sl[sidx][1]]
  u3 = pts3[sl[sidx][2]]

  @test isapprox( u1[1], getPoints(P)[sidx][1])
  @test isapprox( u3[3], getPoints(P)[sidx][3])

end

##
end




#