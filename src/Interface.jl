# Interface

export productbelief

mutable struct ManifoldKernelDensity{M <: MB.AbstractManifold{MB.ℝ}, B <: BallTreeDensity}
  manifold::M
  belief::B
end
const MKD{M,B} = ManifoldKernelDensity{M, B}

ManifoldKernelDensity(mani::M, bel::B) where {M <: MB.AbstractManifold, B <: BallTreeDensity} = MKD{M,B}(mani,bel)


function Base.show(io::IO, mkd::ManifoldKernelDensity{M,B}) where {M, B}
  printstyled(io, "ManifoldKernelDensity{$M,$B}(\n", bold=true )
  show(io, mkd.belief)
  println(io, ")")
end

Base.show(io::IO, ::MIME"text/plain", mkd::ManifoldKernelDensity) = show(io, mkd)

# NOTE, this product does not handle combinations of different partial beliefs properly yet
function *(PP::AbstractVector{<:MKD{M,B}}) where {M<:MB.AbstractManifold{MB.ℝ},B}
  manifoldProduct(PP, PP[1].manifold)
end

function *(P1::MKD{M,B}, P2::MKD{M,B}, P_...) where {M<:MB.AbstractManifold{MB.ℝ},B}
  manifoldProduct([P1;P2;P_...], P1.manifold)
end


## ================================================================================================================================
# Serialization
## ================================================================================================================================

# abstract type JSONManifoldKernelDensity end

# export JSONManifoldKernelDensity

function Base.convert(::Type{<:AbstractString}, 
                      mkd::ManifoldKernelDensity)
  #
  dict = Dict{Symbol, String}()
  dict[:_type] = "ManifoldKernelDensity"
  dict[:belief] = KDE.string( mkd.belief )
  dict[:manifold] = string(mkd.manifold)

  JSON2.write(dict)
end

function Base.convert(::Type{<:ManifoldKernelDensity}, str::AbstractString)
  dict = JSON2.read(str)
  # make module specific
  # good references: 
  #  https://discourse.julialang.org/t/converting-string-to-datatype-with-meta-parse/33024/2
  #  https://discourse.julialang.org/t/is-there-a-way-to-import-modules-with-a-string/15723/6
  manisplit = split(dict[:manifold], '.')
  manimod, manitp = if 1 < length(manisplit)
    modex = Symbol(manisplit[1])
    @eval($modex), dict[:manifold][(length(manisplit[1])+2):end]
  else
    Main, dict[:manifold]
  end
  manip = Meta.parse(manitp)
  manis = Core.eval(manimod, manip) # could not get @eval to work with $
  belief_ = convert(BallTreeDensity, dict[:belief])
  ManifoldKernelDensity(manis, belief_)
end



## ================================================================================================================================
# pass through API
## ================================================================================================================================

# not exported yet
# getManifold(x::ManifoldKernelDensity) = x.manifold


import KernelDensityEstimate: getPoints, getBW, Ndim, Npts, getWeights, marginal 
import KernelDensityEstimate: getKDERange, getKDEMax, getKDEMean, getKDEfit
import KernelDensityEstimate: sample, rand, resample, kld, minkld
import Random: rand

export getPoints, getBW, Ndims, Npts
export getKDERange, getKDEMax, getKDEMean, getKDEfit
export sample, rand, resample, kld, minkld

# with DFG v0.15 change points to Vector{P}
function getPoints(x::ManifoldKernelDensity, w...;kw...) 
  pts = getPoints(x.belief,w...;kw...)
  @cast ptsArr[j][i] := pts[i,j]
  return ptsArr
end


getBW(x::ManifoldKernelDensity, w...;kw...) = getBW(x.belief,w...;kw...)

Ndim(x::ManifoldKernelDensity, w...;kw...) = Ndim(x.belief,w...;kw...)
Npts(x::ManifoldKernelDensity, w...;kw...) = Npts(x.belief,w...;kw...)

getWeights(x::ManifoldKernelDensity, w...;kw...) = getWeights(x.belief, w...;kw...)
# marginal(_)
sample(x::ManifoldKernelDensity, w...;kw...) = sample(x.belief, w...;kw...)
Random.rand(x::ManifoldKernelDensity, d::Integer=1) = rand(x.belief, d)
resample(x::ManifoldKernelDensity, w...;kw...) = resample(x.belief, w...;kw...)

getKDERange(x::ManifoldKernelDensity, w...;kw...) = getKDERange(x.belief, w...;kw...)
getKDEMax(x::ManifoldKernelDensity, w...;kw...) = getKDEMax(x.belief, w...;kw...)
getKDEMean(x::ManifoldKernelDensity, w...;kw...) = getKDEMean(x.belief, w...;kw...)
getKDEfit(x::ManifoldKernelDensity, w...;kw...) = getKDEfit(x.belief, w...;kw...)

kld(x::ManifoldKernelDensity, w...;kw...) = kld(x.belief, w...;kw...)
minkld(x::ManifoldKernelDensity, w...;kw...) = minkld(x.belief, w...;kw...)

(x::ManifoldKernelDensity)(w...;kw...) = x.belief(w...;kw...)



## ================================================================================================================================
# Legacy Interface for product of full and partial dimensions
## ================================================================================================================================

# legacy




function _partialProducts!( pGM::AbstractVector{P}, 
                            partials, 
                            manifold::MB.AbstractManifold; 
                            inclFull::Bool=true  ) where P
  #
  manis = convert(Tuple, manifold)

  # FIXME, remove temporary Tuple manifolds method 
  for (dimnum,pp) in partials
    dimv = [dimnum...]
    # include previous calcs (if full density products were done before partials)
    # NOTE, SWAPPED LOGIC ORDER OF inclFull from previous code
    if inclFull
      partialMani = _buildManifoldPartial(manifold, dimv)
      partialPts = [pGM[i][dimv] for i in 1:length(pGM)]
      push!( pp, AMP.manikde!(partialPts, partialMani) )
    end
    # take product of this partial's subset of dimensions
    partial_GM = AMP.manifoldProduct(pp, manifold, Niter=1, partialDimsWorkaround=dimv) |> getPoints
    # partial_GM = AMP.manifoldProduct(pp, (manis[dimv]...,), Niter=1) |> getPoints

    for i in 1:length(pGM)
      pGM[i][dimv] = partial_GM[i]
    end
  end
  nothing
end


"""
    $SIGNATURES

Take product of `dens` accompanied by optional `partials` proposal belief densities.

Notes
-----
- `d` dimensional product approximation
- `partials` are treated per each unique Tuple subgrouping, i.e. (1,2), (2,), ...
- Incorporate ApproxManifoldProducts to process variables in individual batches.

DevNotes
- TODO Consolidate with AMP.manifoldProduct, especially concerning partials. 
"""
function productbelief( denspts::AbstractVector{P},
                        manifold::MB.AbstractManifold,
                        dens::Vector{<:ManifoldKernelDensity},
                        partials::Dict{Any, <:AbstractVector{<:ManifoldKernelDensity}},
                        N::Int;
                        dbg::Bool=false,
                        logger=ConsoleLogger()  ) where P <: AbstractVector{<:Real}
  #
  # TODO only works of P <: Vector
  lennonp = length(dens)
  lenpart = length(partials)
  Ndims = size(denspts[1])
  with_logger(logger) do
    @info "[$(lennonp)x$(lenpart)p,d$(Ndims),N$(N)],"
  end
  
  # # resize for #1013
  # if size(denspts,2) < N
  #   pGM = zeros(size(denspts,1),N)
  #   pGM[:,1:size(denspts,2)] .= denspts
  # else
  #   pGM = deepcopy(denspts)
  # end
  
  # new, slightly condensed partialProduct operation
  # TODO VALIDATE inclFull is the right order
  (pGM, inclFull) = if 0 < lennonp
    getPoints(AMP.manifoldProduct(dens, manifold, Niter=1)), true # false
  elseif lennonp == 0 && 0 < lenpart
    deepcopy(denspts), false # true
  else
    error("Unknown density product lennonp=$(lennonp), lenpart=$(lenpart)")
  end

  # take the product between partial dimensions
  _partialProducts!(pGM, partials, manifold, inclFull=inclFull)

  return pGM
end




#