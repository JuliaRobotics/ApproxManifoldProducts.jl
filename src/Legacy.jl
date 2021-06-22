# legacy content to facilitate transition to AMP


"""
    $SIGNATURES

Lots to do here, see RoME.jl #244 and standardized usage with Manifolds.jl.

Notes
- diffop( test, reference )   <===>   ΔX = inverse(test) * reference

DevNotes
- FIXME replace with Manifolds.jl #41, RoME.jl #244
"""
function buildHybridManifoldCallbacks(manif::Tuple)
  # TODO use multiple dispatch instead -- will be done for second version of system
  addopT = []
  diffopT = []
  getManiMu = []
  getManiLam = []

  for mn in manif
    if mn == :Euclid
      push!(addopT, +)
      push!(diffopT, -)
      push!(getManiMu, KDE.getEuclidMu)
      push!(getManiLam, KDE.getEuclidLambda)
    elseif mn == :Circular
      push!(addopT, addtheta)
      push!(diffopT, difftheta)
      push!(getManiMu, getCircMu)
      push!(getManiLam, getCircLambda)
    else
      error("Unrecognized manifold $(mn)")
    end
  end

  return (addopT...,), (diffopT...,), (getManiMu...,), (getManiLam...,)
end

# FIXME temp conversion during consolidation
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(Euclid)}) = (:Euclid,)
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(Euclid2)}) = (:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(Euclid3)}) = (:Euclid,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(Euclid4)}) = (:Euclid,:Euclid,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(SE2_Manifold)}) = (:Euclid,:Euclid,:Circular)
Base.convert(::Type{<:Tuple}, ::Type{<: typeof(SE3_Manifold)}) = (:Euclid,:Euclid,:Euclid,:Circular,:Circular,:Circular)

Base.convert(::Type{<:Tuple}, ::typeof(Euclid)) = (:Euclid,)
Base.convert(::Type{<:Tuple}, ::typeof(Euclid2)) = (:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::typeof(Euclid3)) = (:Euclid,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::typeof(Euclid4)) = (:Euclid,:Euclid,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, ::typeof(SE2_Manifold)) = (:Euclid,:Euclid,:Circular)
Base.convert(::Type{<:Tuple}, ::typeof(SE3_Manifold)) = (:Euclid,:Euclid,:Euclid,:Circular,:Circular,:Circular)


"""
    $(SIGNATURES)

Calculate the KDE bandwidths for each dimension independly, as per manifold of each.  Return vector of all dimension bandwidths.
"""
function getKDEManifoldBandwidths(pts::AbstractMatrix{<:Real},
                                  manif::T1 ) where {T1 <: Tuple}
  #
  ndims = size(pts, 1)
  bws = ones(ndims)

  for i in 1:ndims
    if manif[i] == :Euclid
      bws[i] = getBW( kde!(pts[i,:]) )[1,1]
    elseif manif[i] == :Circular
      bws[i] = getBW( kde!_CircularNaiveCV( pts[i,:] ) )[1,1]
    else
      error("Unrecognized manifold $(manif[i])")
    end
  end

  return bws
end

function ensurePeriodicDomains!( pts::AA, manif::T1 ) where {AA <: AbstractArray{Float64,2}, T1 <: Tuple}

  i = 0
  for mn in manif
    i += 1
    if manif[i] == :Circular
      pts[i,:] = TUs.wrapRad.(pts[i,:])
    end
  end

  nothing
end



manikde!( ptsArr::AbstractVector{P}, 
          M::MB.AbstractManifold  ) where P <: AbstractVector = ManifoldKernelDensity(M, ptsArr) 
#

manikde!( ptsArr::AbstractVector{P}, 
          bw::AbstractVector{<:Real}, 
          M::MB.AbstractManifold  ) where P <: AbstractVector = ManifoldKernelDensity(M, ptsArr, bw)
#

#