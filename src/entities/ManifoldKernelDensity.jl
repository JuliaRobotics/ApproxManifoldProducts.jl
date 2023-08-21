

struct ManifoldKernelDensity{M <: MB.AbstractManifold, B <: BallTreeDensity, L, P <: Union{<:AbstractVector, <:ArrayPartition}}
  manifold::M
  """ legacy expects matrix of coordinates (as columns) """
  belief::B
  _partial::L
  """ just an example point for local access to the point data type"""
  _u0::P
  infoPerCoord::Vector{Float64}
end
const MKD{M,B,L} = ManifoldKernelDensity{M, B, L}


