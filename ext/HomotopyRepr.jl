
using DistributedFactorGraphs

import DistributedFactorGraphs: getManifold


export 
  AbstractPartialTraits,
  AbstractHomotopyTruncation,
  MajorMaxDepth

export 
  getStateType, 
  getManifold, 
  getReprType, 
  getTruncation, 
  getPartial


# Forward looking abstract for DFG v1.x development of partials using traits, but not yet implemented
abstract type AbstractPartialTraits end

abstract type AbstractHomotopyTruncation end

abstract type AbstractDensityBasis end

"""
HomotopyRepresentation is a struct that encapsulates the representation of a homotopy density.
  HomotopyDensity is a very broad and 99% agnostic serialization type whose method implementations should dispatch
  on the type of the representation, which is expected to be a concrete type with necessary dispatch info.

Comment on future-proofing: at time of writing (26Q2), we anticipate a long and methodic development of 
  hybrid-(non)parametric computational methods which can all fit in the same general LTS framework (i.e. DFG v1).

Future-proofing is achieved by making the representation type a concrete struct which is JSON.jl compliant, barr
- elementary lift and lower implementations for HomotopyRepresentation.

Comments on type parameters:
- statetype is the type of the state, 
  which is either a Manifolds.jl manifold type or a DFG statetype
- L is the type of the partial, future expectation is for improved traits-based partials while,
 legacy used tuples to specify coord dims.
- reprtype is the type of the density basis, 
  e.g. ConcentratedGaussianKernel and is expected to evolve into continuous eigen vectors and wavelets.
- truncation type relates to model order reduction technique embedded in the HomotopyDensity, 
  e.g. MajorMaxDepth is a simple binary tree truncation with N major levels. 
"""
struct HomotopyRepresentation{
  statetype <: Union{<:AbstractManifold, <:DistributedFactorGraphs.AbstractStateType},
  L <: AbstractPartialLegacyCompat, # Future use <:AbstractPartialTraits
  reprtype <: AbstractDensityBasis, 
  truncation <: AbstractHomotopyTruncation
} 
  """ 
  Used for either DistributedFactorGraphs statetype or Manifolds.jl manifold type, depending on context. 
  Expect official support for serde only for DFG statetypes.  Note, DFG statetypes are built on top of Manifolds.jl.
  Use `getManifold(::HomotopyRepresentation)` to get the manifold type regardless of context.

  Note, and explicit object `_statekind` is needed for the Manifolds.jl only context, but note this field is not serialized
  """
  statekind_noserde::statetype
  """
  Future of partials is to use traits, so an abstract type is warranted.
  Legacy is object of either Nothing, Tuple, or Vector{Int}, but something better is needed
  """
  partial::L
end


struct MajorMaxDepth{
  N
} <: AbstractHomotopyTruncation end



getMajorsLength(::Type{<:AbstractHomotopyTruncation}) = 1
getMajorsLength(repr::HomotopyRepresentation) = getMajorsLength(getTruncation(repr))

# Binary tree with N major levels
getMajorsLength(::Type{MajorMaxDepth{N}}) where {N} = N^2 - 1

# trivial case -- should be in DFG instead FIXME
getManifold(manif::AbstractManifold) = manif

getStateType(::HomotopyRepresentation{statetype}) where {statetype} = statetype
getPartial(hr::HomotopyRepresentation) = hr.partial
getReprType(::HomotopyRepresentation{S,L,reprtype}) where {S, L, reprtype} = reprtype
getTruncation(::HomotopyRepresentation{S,L,R,truncation}) where {S,L,R,truncation} = truncation
# supports both DFG and ManifoldsBase
getManifold(repr::HomotopyRepresentation) = getManifold(repr.statekind_noserde) 


