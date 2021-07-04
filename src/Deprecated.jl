

## ======================================================================================================
## Remove below before v0.5
## ======================================================================================================

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