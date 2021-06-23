# legacy content to facilitate transition to AMP


function _reducePartialManifoldElements(el::Symbol)
  if el == :Euclid
    return Euclidean(1)
  elseif el == :Circular
    return Circle()
  end
  error("unknown manifold_symbol $el")
end

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

function ManifoldKernelDensity( M::MB.AbstractManifold,
                                ptsArr::AbstractVector{P},
                                bw::Union{<:AbstractVector{<:Real},Nothing}=nothing  ) where P
  #
  # FIXME obsolete
  arr = Matrix{Float64}(undef, length(ptsArr[1]), length(ptsArr))
  @cast arr[i,j] = ptsArr[j][i]
  manis = convert(Tuple, M)
  # find or have the bandwidth
  _bw = bw === nothing ? getKDEManifoldBandwidths(arr, manis ) : bw
  addopT, diffopT, _, _ = buildHybridManifoldCallbacks(manis)
  bel = KernelDensityEstimate.kde!(arr, _bw, addopT, diffopT)
  return ManifoldKernelDensity(M, bel)
end

# override
function marginal(x::ManifoldKernelDensity, dims, w...;kw...)
  manis = convert(Tuple, x.manifold)
  partMani = _reducePartialManifoldElements(manis[dims])
  ManifoldKernelDensity(partMani, marginal(x.belief, dims, w...;kw...))
end

# internal workaround function for building partial submanifold dimensions, must be upgraded/standarized
function _buildManifoldPartial( fullM::MB.AbstractManifold, 
                                partial_coord_dims )
  #
  # temporary workaround during Manifolds.jl integration
  manif = convert(Tuple, fullM)[partial_coord_dims]
  # 
  newMani = MB.AbstractManifold[]
  for me in manif
    push!(newMani, _reducePartialManifoldElements(me))
  end

  # assume independent dimensions for definition, ONLY USED AS DECORATOR AT THIS TIME, FIXME
  return ProductManifold(newMani...)
end






#