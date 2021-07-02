

## ======================================================================================================
## Remove below before v0.5
## ======================================================================================================

@deprecate manikde!( vecP::AbstractVector, M::MB.AbstractManifold ) manikde!(M, vecP)
@deprecate manikde!( vecP::AbstractVector, bw::AbstractVector{<:Real}, M::MB.AbstractManifold ) manikde!(M, vecP, bw=bw) 

# function ManifoldKernelDensity( M::MB.AbstractManifold, 
#                                 ptsArr::AbstractVector{P} ) where P
#   #
#   # FIXME obsolete
#   arr = Matrix{Float64}(undef, length(ptsArr[1]), length(ptsArr))
#   @cast arr[i,j] = ptsArr[j][i]
#   manis = convert(Tuple, M)
#   bw = getKDEManifoldBandwidths(arr, manis )
#   addopT, diffopT, _, _ = buildHybridManifoldCallbacks(manis)
#   bel = KernelDensityEstimate.kde!(arr, bw, addopT, diffopT)
#   return ManifoldKernelDensity(M, bel)
# end


export ensurePeriodicDomains!
function ensurePeriodicDomains!( pts::AA, manif::T1 ) where {AA <: AbstractArray{Float64,2}, T1 <: Tuple}
  @warn "ensurePeriodicDomains! is being deprecated without replacement."
  i = 0
  for mn in manif
    i += 1
    if manif[i] == :Circular
      pts[i,:] = TUs.wrapRad.(pts[i,:])
    end
  end

  nothing
end

function manifoldProduct( ff::Union{Vector{BallTreeDensity},<:Vector{<:ManifoldKernelDensity}},
  manis::Tuple;
  kwargs... )
#
error("Obsolete, use manifoldProduct(::Vector{MKD}, <:AbstractManifold) instead.\n`::NTuple{Symbol}` for manifolds is outdated, use `getManifold(...)` and `ManfoldsBase.AbstractManifold` instead.")
end

@deprecate getManifolds(::Type{<:T}) where {T <: ManifoldsBase.AbstractManifold} convert(Tuple, T)
getManifolds(::T) where {T <: ManifoldsBase.AbstractManifold} = getManifolds(T)


# struct Euclid2 <: MB.AbstractManifold{MB.ℝ}
#   dof::Int
#   addop::Function
#   diffop::Function
#   getMu
#   getLambda
#   domain::Tuple{Tuple{Float64,Float64},Tuple{Float64, Float64}}
# end

# Euclid2() = Euclid2(2, +, -, KDE.getEuclidMu, KDE.getEuclidLambda, ((-Inf,Inf),(-Inf,Inf)))


# function *(PP::AbstractVector{<:MKD{typeof(EuclideanManifold),BallTreeDensity}})
#   bds = (p->p.belief).(PP)
#   *(bds)
# end

# """
#     $(SIGNATURES)

# Legacy extension of KDE.kde! function to approximate smooth functions based on samples, using likelihood cross validation for bandwidth selection.  This method allows approximation over hybrid manifolds.
# """
# function manikde!(pts::AbstractMatrix{<:Real},
#                   bws::AbstractVector{<:Real},
#                   manifolds::T  ) where {T <: Tuple}
#   #
#   error("NTuple{Symbol} for manifolds definition is now obsolete")
#   addopT, diffopT, _, _ = buildHybridManifoldCallbacks(manifolds)
#   @show size(pts), typeof(pts)
#   KernelDensityEstimate.kde!(pts, bws, addopT, diffopT)
# end

# function manikde!(ptsArr::AbstractVector{P},
#                   bws::AbstractVector{<:Real},
#                   manifolds::T  ) where {P <: AbstractVector, T <: Tuple}
#   #
#   error("NTuple{Symbol} for manifolds definition is now obsolete")
#   pts = Array{Float64,2}(undef, length(ptsArr[1]),length(ptsArr))
#   @cast pts[i,j] = ptsArr[j][i]
#   manikde!(pts, bws, manifolds)
# end

# function manikde!(pts::AbstractMatrix{<:Real},
#                   manifolds::T  ) where {T <: Tuple}
#   #
#   error("NTuple{Symbol} for manifolds definition is now obsolete")
#   bws = getKDEManifoldBandwidths(pts, manifolds)
#   ensurePeriodicDomains!(pts, manifolds)
#   AMP.manikde!(pts, bws, manifolds)
# end

# function manikde!(ptsArr::AbstractVector{P},
#                   manifolds::T  ) where {P <: AbstractVector, T <: Tuple}
#   #
#   error("NTuple{Symbol} for manifolds definition is now obsolete")
#   pts = Array{Float64,2}(undef, length(ptsArr[1]),length(ptsArr))
#   @cast pts[i,j] = ptsArr[j][i]
#   manikde!(pts, manifolds)
# end

@deprecate ManifoldBelief(w...;kw...) ManifoldKernelDensity(w...;kw...)
function ManifoldBelief(::M, mkd::ManifoldKernelDensity{M,T}) where {M <: MB.AbstractManifold{MB.ℝ}, T} 
  @warn "ManifoldBelief is deprecated, use ManifoldKernelDensity instead"
  return mkd
end