# Test naive implementation of entropy calculations towards efficient calculation of entropy on a manifold


export
  logmap_SO2,
  difftheta,
  addtheta,
  rbfAccAt!,
  rbf!,
  rbf,
  evaluateManifoldNaive1D!,
  manifoldLooCrossValidation,
  kde!_CircularNaiveCV,
  getCircMu,
  getCircLambda,
  logmap_SO2,
  difftheta,
  addtheta


global const reci_s2pi=1.0/sqrt(2.0*pi) # 1.0/2.5066282746310002


# On-manifold circular product callbacks

# manifold distance, add, and subtract
function logmap_SO2(Rl::Matrix{Float64})
  ct = abs(Rl[1,1]) > 1.0 ? 1.0 : Rl[1,1]  # reinserting the sign below
  -sign(Rl[2,1])*acos(ct)
end
difftheta(wth1, wth2)::Float64 = logmap_SO2(TUs.R(wth1)'*TUs.R(wth2))
addtheta(wth1, wth2)::Float64 = TUs.wrapRad( wth1+wth2 )

# manifold get Gaussian products mean
getCircMu = (m::Vector{Float64}, s::Vector{Float64}, dummy::Float64) -> TUs.wrapRad(get2DMu(m, s, diffop=difftheta, initrange=(-pi+0.0,pi+0.0)))
# getCircMu = (m::Vector{Float64}, s::Vector{Float64}, dummy::Float64) -> TUs.wrapRad(get2DMuMin(m, s, diffop=difftheta, initrange=(-pi+0.0,pi+0.0)))

getCircLambda(x) = getEuclidLambda(x)


"""
    $SIGNATURES

Probability density function `p(x)`, as estimated by kernels
```math
hatp_{-j}(x) = 1/(N-1) Σ_{i != j}^N frac{1}{sqrt{2pi}σ } exp{ -frac{(x-μ)^2}{2 σ^2} }
```
"""
function normDistAccAt!(ret::AV,
                        idx::Int,
                        x::Float64,
                        sigma::Float64,
                        w::Float64=1.0  )::Nothing where {AV <: AbstractVector}
    global reci_s2pi
    @fastmath ret[idx] += w*reci_s2pi/sigma * exp( - (x^2)/(2.0*(sigma^2)) )
    return nothing
end

function rbfAccAt!(ret::AV,
                   idx::Int,
                   x::Float64,
                   μ::Float64=0.0,
                   σ::Float64=1.0,
                   w::Float64=1.0,
                   diffop::Function=-)::Nothing  where {AV <: AbstractVector}
    #
    normDistAccAt!(ret, idx, diffop(x, μ), σ, w)
    nothing
end
function rbf!(ret::AV,
              x::Float64,
              μ::Float64=0.0,
              σ::Float64=1.0,
              diffop::Function=-)::Nothing  where {AV <: AbstractVector}
    #
    ret[1] = 0.0
    normDistAccAt!(ret, 1, diffop(x,μ), σ)
    nothing
end


function rbf(x::Float64, μ::Float64=0.0, σ::Float64=1.0)
    ret = Vector{Float64}(undef, 1) # initialized in rbf!(..)
    rbf!(ret, x, μ, σ)
    return ret[1]
end


"""
    $SIGNATURES

Evalute the KDE naively as equally weighted Gaussian kernels with common bandwidth.
This function does, however, allow on-manifold evaluations.
"""
function evaluateManifoldNaive1D!(ret::Vector{Float64},
                                  idx::Int,
                                  pts::Array{Float64,1},
                                  bw::Float64,
                                  x::Array{Float64,1},
                                  loo::Int=-1,
                                  diffop=-  )::Nothing
    #
    dontskip = loo == -1
    N = length(pts)
    reci_N = dontskip ? 1.0/N : 1.0/(N-1)
    for j in 1:N
        if dontskip || loo != j
            manifolddist = diffop(pts[loo], pts[j])
            normDistAccAt!(ret, idx, manifolddist, bw, reci_N)
        end
    end

    return nothing
end
function evaluateManifoldNaive1D!(ret::Vector{Float64},
                                  idx::Int,
                                  bd::BallTreeDensity,
                                  x::Array{Float64,1},
                                  loo::Int=-1,
                                  diffop=-  )::Nothing
    #
    evaluateManifoldNaive1D!(ret, idx, getPoints(bd)[:], getBW(bd)[1,1], x, loo, diffop )
end

"""
    $SIGNATURES

Calculate negative entropy with leave one out (j'th element) cross validation.

Background
==========

From: Silverman, B.: Density Estimation for Statistics and Data Analysis, 1986, p.52

Probability density function `p(x)`, as estimated by kernels
```math
hatp_{-j}(x) = 1/(N-1) Σ_{i != j}^N frac{1}{sqrt{2pi}σ } exp{ -frac{(x-μ)^2}{2 σ^2} }
```
and has Cross Validation number as the average log evaluations of leave one out `hatp_{-j}(x)`:
```math
CV(p) = 1/N Σ_i^N log hat{p}_{-j}(x_i)
```

This quantity `CV` is related to an entropy `H(p)` estimate via:
```math
H(p) = -CV(p)
```
"""
function manifoldLooCrossValidation(pts::Array,
                            bw::Float64;
                            own::Bool=true,
                            diffop::Function=-  )
    #
    N = maximum(size(pts))
    h = [bw;]
    loo = zeros(N)
    @inbounds for i in 1:N
        if !own
            # validation version
            pts99 = pts[[1:(i-1);(i+1):end]]
            p99 = kde!(pts99, h)
            loo[i] = log(p99([pts[i];])[1])
        else
            # own naive entropy calculation
            loo[i] = 0.0
            evaluateManifoldNaive1D!(loo, i, pts, bw, pts, i, diffop)
            loo[i] = log(loo[i])
        end
    end
    return  sum(loo)/N
end


function kde!_CircularNaiveCV(points::A) where {A <: AbstractArray{Float64,1}}

  # initial setup parameters
  dims = 1 # size(points,1)
  bwds = zeros(dims)
  # initial testing values
  lower = 0.001
  upper = 2pi

  # excessive for loop for leave one out likelihiood cross validation (Silverman 1986, p.52)
  for i in 1:dims
    minEntropyLOOCV = (bw) -> -manifoldLooCrossValidation(points, bw, own=true, diffop=difftheta)
    res = optimize(minEntropyLOOCV, lower, upper, GoldenSection(), x_tol=0.001)
    bwds[i] = res.minimizer
  end

  # cosntruct the kde with CV optimized bandwidth
  p = kde!( points, bwds, (addtheta,), (difftheta,) )

  return p
end


#
