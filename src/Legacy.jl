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
Base.convert(::Type{<:Tuple}, mani::Type{<: Euclid}) = (:Euclid,)
Base.convert(::Type{<:Tuple}, mani::Type{<: Euclid2}) = (:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, mani::Type{<: Euclid3}) = (:Euclid,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, mani::Type{<: Euclid4}) = (:Euclid,:Euclid,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, mani::Type{<: SE2_Manifold}) = (:Euclid,:Euclid,:Circular)
Base.convert(::Type{<:Tuple}, mani::Type{<: SE2E2_Manifold}) = (:Euclid,:Euclid,:Circular,:Euclid,:Euclid)
Base.convert(::Type{<:Tuple}, mani::Type{<: SE3_Manifold}) = (:Euclid,:Euclid,:Euclid,:Circular,:Circular,:Circular)



"""
    $(SIGNATURES)

Calculate the KDE bandwidths for each dimension independly, as per manifold of each.  Return vector of all dimension bandwidths.
"""
function getKDEManifoldBandwidths(pts::AA,
                                  manif::T1 ) where {AA <: AbstractArray{Float64,2}, T1 <: Tuple}
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


"""
    $(SIGNATURES)

Legacy extension of KDE.kde! function to approximate smooth functions based on samples, using likelihood cross validation for bandwidth selection.  This method allows approximation over hybrid manifolds.
"""
function manikde!(pts::AA2,
                  bws::Vector{Float64},
                  manifolds::T  ) where {AA2 <: AbstractArray{Float64,2}, T <: Tuple}
  #
  addopT, diffopT, getManiMu, getManiLam = buildHybridManifoldCallbacks(manifolds)
  bel = KernelDensityEstimate.kde!(pts, bws, addopT, diffopT)
end

function manikde!(pts::AA2,
                  manifolds::T  ) where {AA2 <: AbstractArray{Float64,2}, T <: Tuple}
  #
  bws = getKDEManifoldBandwidths(pts, manifolds)
  ensurePeriodicDomains!(pts, manifolds)
  ApproxManifoldProducts.manikde!(pts, bws, manifolds)
end

function manikde!(pts::AA2,
                  manifold::Type{<:MB.Manifold{MB.ℝ}}  ) where {AA2 <: AbstractArray{Float64,2}}
  #
  maniT = convert(Tuple, manifold)
  manikde!(pts, maniT)
end