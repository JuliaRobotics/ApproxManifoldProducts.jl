
## ======================================================================================================
## Remove below before v0.7
## ======================================================================================================


## New Manifolds.jl aware API -- TODO find the right file placement



# # TODO, hack, use the proper Manifolds.jl intended vectoration methods instead
# _makeVectorManifold(::MB.AbstractManifold, arr::AbstractArray{<:Real}) = arr
# _makeVectorManifold(::MB.AbstractManifold, val::Real) = [val;]
# _makeVectorManifold(::M, prr::ProductRepr) where {M <: typeof(SpecialEuclidean(2))} = coords(M, prr)
# _makeVectorManifold(::M, prr::ProductRepr) where {M <: typeof(SpecialEuclidean(3))} = coords(M, prr)



## ======================================================================================================
## Remove below before v0.6
## ======================================================================================================

@deprecate setPointsManiPartial!( Mdest::AbstractManifold, 
                                  dest, 
                                  Msrc::AbstractManifold, 
                                  src, 
                                  partial::AbstractVector{<:Integer},
                                  asPartial::Bool=true ) setPointPartial!( Mdest, dest, Msrc, src, partial, asPartial )


export productbelief

"""
    $SIGNATURES

Take product of `dens` (including optional partials beliefs) as proposals to be multiplied together.

Notes
-----
- Return points of full dimension, even if only partial dimensions in proposals.
  - 'Other' dimensions left unchanged from incoming `denspts`
- `d` dimensional product approximation
- Incorporate ApproxManifoldProducts to process variables in individual batches.

DevNotes
- TODO Consolidate with [`AMP.manifoldProduct`](@ref), especially concerning partials. 
"""
function productbelief( denspts::AbstractVector{P},
                        manifold::MB.AbstractManifold,
                        dens::Vector{<:ManifoldKernelDensity},
                        N::Int;
                        asPartial::Bool=false,
                        dbg::Bool=false,
                        logger=ConsoleLogger()  ) where P
  #

  @warn "productbelief is being deprecated, use manifoldProduct together with getPoints instead."
  mkd = AMP.manifoldProduct(dens, manifold, Niter=1, oldPoints=denspts, logger=logger)
  pGM = getPoints(mkd, asPartial)

  return pGM
end


# function calcMean(mkd::ManifoldKernelDensity{M}) where {M <: ManifoldsBase.AbstractManifold}
#   data = getPoints(mkd)
#   # Returns the mean point on manifold for consitency
#   mean(mkd.manifold, data)  
# end


@deprecate calcVariableCovarianceBasic(M::AbstractManifold, vecP::Vector{P}) where P calcCovarianceBasic(M, vecP)

## ======================================================================================================
## Remove below before v0.5
## ======================================================================================================

#
# """
#     $SIGGNATURES
#
# Assemble oplus and ominus operations from given manifols.
#
# Related
#
# buildHybridManifoldCallbacks
# """
# function getManifoldOperations(manis::T) where {T <: Tuple}
#
# end
#

uncoords(::Type{<:MB.AbstractManifold}, w...; kw...) = error("uncoords(::Type{<:MB.AbstractManifold}, w...; kw...) is obsolete, use makePointFromCoords(M::MB.AbstractManifold, w...) instead.")

coords(::Type{<:MB.AbstractManifold}, w...;   kw...) = error("coords(::Type{<:typeof(SpecialEuclidean(2))}, w...; kw...) is obsolete, use makeCoordsFromPoint(M::MB.AbstractManifold, w...) instead.")


# coords(::Type{<:typeof(SpecialEuclidean(2))}, p::ProductRepr) = [p.parts[1][1], p.parts[1][2], atan(p.parts[2][2,1],p.parts[2][1,1])]

# function coords(::Type{<:typeof(SpecialEuclidean(3))}, p::ProductRepr)
#   wELo = TU.convert(Euler, SO3(p.parts[2]))
#   [p.parts[1][1:3]; wELo.R; wELo.P; wELo.Y]
# end

# function uncoords(::Type{<:typeof(SpecialEuclidean(2))}, p::AbstractVector{<:Real}, static::Bool=true)
#   α = p[3] 
#   ArrConst = static ? SA : eltype(α)
#   return ProductRepr((ArrConst[p[1], p[2]]), ArrConst[cos(α) -sin(α); sin(α) cos(α)])
# end
# # function uncoords(::Type{<:typeof(SpecialEuclidean(2))}, p::AbstractVector{<:Real})
# #   α = p[3]
# #   return ProductRepr(([p[1], p[2]]), [cos(α) -sin(α); sin(α) cos(α)])
# # end

# function uncoords(::Type{<:typeof(SpecialEuclidean(3))}, p::AbstractVector{<:Real})
#   # α = p[3]
#   wRo = TU.convert(SO3, Euler(p[4:6]...))
#   return ProductRepr(([p[1], p[2], p[3]]), wRo.R)
# end



function getPointsManifold(mkd::ManifoldKernelDensity{M}) where {M <: Euclidean}
  @warn "getPointsManifold is being deprecated, use getPoints(::MKD)::Vector{P} instead"
  data_ = getPoints(mkd.belief)
  TensorCast.@cast data[i][j] := data_[j,i]
  return data
end

function getPointsManifold(mkd::ManifoldKernelDensity{M}) where {M <: Circle}
  @warn "getPointsManifold is being deprecated, use getPoints(::MKD)::Vector{P} instead"
  data_ = getPoints(mkd.belief)
  return data_[:]
end

function getPointsManifold(mkd::ManifoldKernelDensity{M}) where {M <: SpecialEuclidean}
  @warn "getPointsManifold is being deprecated, use getPoints(::MKD)::Vector{P} instead"
  data_ = getPoints(mkd.belief)
  [uncoords(M, view(data_, :, i)) for i in 1:size(data_,2)]
end


@deprecate mmd!(v::AbstractVector{<:Real}, a::AbstractArray,b::AbstractArray,MF::MB.AbstractManifold, w...; kw...) mmd!(MF, v, a, b, w...; kw...)
@deprecate mmd(a::AbstractArray,b::AbstractArray,MF::MB.AbstractManifold, w...; kw...) mmd(MF, a, b, w...; kw...)


# function ker( ::typeof(Euclidean(1)),
#               x::AbstractVector{P1},
#               y::AbstractVector{P2},
#               dx::Vector{<:Real},
#               i::Int,
#               j::Int;
#               sigma::Real=0.001 ) where {P1<:AbstractVector, P2<:AbstractVector}
#   #
#   dx[1] = x[i][1]
#   dx[1] -= y[j][1]
#   dx[1] *= dx[1]
#   dx[1] *= -sigma
#   exp( dx[1] )
# end

# function ker( ::typeof(Euclidean(2)),
#               x::AbstractVector{P1},
#               y::AbstractVector{P2},
#               dx::Vector{<:Real},
#               i::Int,
#               j::Int;
#               sigma::Real=0.001 ) where {P1<:AbstractVector, P2<:AbstractVector}
#   #
#   dx[1] = x[i][1]
#   dx[2] = x[i][2]
#   dx[1] -= y[j][1]
#   dx[2] -= y[j][2]
#   dx .^= 2
#   dx[1] += dx[2]
#   dx[1] *= -sigma
#   exp( dx[1] )
# end

# function ker( ::typeof(SE2_Manifold),
#               x::AbstractVector{P1},
#               y::AbstractVector{P2},
#               dx::Vector{<:Real},
#               i::Int,
#               j::Int;
#               sigma::Real=0.001  ) where {P1<:AbstractVector, P2<:AbstractVector}
#   #
#   innov = se2vee(SE2(x[i][:])\SE2(y[j][:]))
#   exp( -sigma*(  innov'*innov  ) )
# end

# # This functin is still very slow, needs speedup
# # Obviously want to get away from the Euler angles throughout
# function ker( ::typeof(SE3_Manifold),
#               x::AbstractVector{P1},
#               y::AbstractVector{P2},
#               dx::Vector{<:Real},
#               i::Int,
#               j::Int;
#               sigma::Real=0.001  )  where {P1<:AbstractVector, P2<:AbstractVector}
#   #
#   innov = veeEuler(SE3(x[i][1:3],Euler((x[i][4:6])...))\SE3(y[j][1:3],Euler((y[j][4:6])...)))
#   exp( -sigma*(  innov'*innov  ) )
# end

# use Circle1 instead
# struct Circular <: MB.AbstractManifold{MB.ℝ}
#   dof::Int
#   addop::Function
#   diffop::Function
#   getMu
#   getLambda
#   domain::Tuple{Float64, Float64}
# end

# Circular() = Circular(1,
#                       addtheta,
#                       difftheta,
#                       getCircMu,
#                       getCircLambda,
#                       (-pi+0.0,pi-1e-15))

# struct SO2Manifold <: MB.AbstractManifold
# end
#
#
# # should not be defined in AMP, since we want IIF indepent of manifolds
# function *(PP::Vector{MKD{SO2Manifold,B}}) where B
#   @info "taking manifold product of $(length(PP)) terms"
#   @warn "SO2Manifold: work in progress"
# end
#
# mbr1 = ManifoldKernelDensity(SO2Manifold, 0.0)
# mbr2 = ManifoldKernelDensity(SO2Manifold, 0.0)
#
# *([mbr1;mbr2])


# # take the full pGM in, but only update the coordinate dimensions that are actually affected by new information.
# function _partialProducts!( pGM::AbstractVector{P}, 
#                             partials::Dict{Any, <:AbstractVector{<:ManifoldKernelDensity}},
#                             manifold::MB.AbstractManifold; 
#                             inclFull::Bool=true  ) where P <: AbstractVector
#   #
#   # manis = convert(Tuple, manifold)
#   keepold = inclFull ? deepcopy(pGM) : typeof(pGM)()

#   # TODO remove requirement for P <: AbstractVector
#   allPartDimsMask = 0 .== zeros(Int, length(pGM[1]))
#   # FIXME, remove temporary Tuple manifolds method 
#   for (dimnum,pp) in partials
#     # mark dimensions getting partial information
#     for d in dimnum
#       allPartDimsMask[d] = true
#     end
#     # change to vector
#     dimv = [dimnum...]
#     # include previous calcs (if full density products were done before partials)
#     partialMani = _buildManifoldPartial(manifold, dimv)
#     # take product of this partial's subset of dimensions
#     partial_GM = AMP.manifoldProduct(pp, partialMani, Niter=1) |> getPoints
#     # partial_GM = AMP.manifoldProduct(pp, (manis[dimv]...,), Niter=1) |> getPoints
    
#     for i in 1:length(pGM)
#       pGM[i][dimv] = partial_GM[i]
#     end
#   end
  
#   # multiply together previous full dim and new product of various partials
#   if inclFull
#     partialPts = [pGM[i][dimv] for i in 1:length(pGM)]
#     push!( pp, AMP.manikde!(partialPts, partialMani) )
#   end
  

#   nothing
# end


@deprecate manikde!( vecP::AbstractVector, M::MB.AbstractManifold ) manikde!(M, vecP)
@deprecate manikde!( vecP::AbstractVector, bw::AbstractVector{<:Real}, M::MB.AbstractManifold ) manikde!(M, vecP, bw) 

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