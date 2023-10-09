

"""
    $TYPEDEF

On-manifold kernel density belief.

Notes
- Allows partials as identified by list of coordinate dimensions e.g. `._partial = [1;3]`
  - When building a partial belief, use full points with necessary information in the specified partial coords.

DevNotes
- WIP AMP issue 41, use generic retractions during manifold products.
"""
struct ManifoldKernelDensity{M <: MB.AbstractManifold, B <: BallTreeDensity, L, P <: AbstractArray}
  manifold::M
  """ legacy expects matrix of coordinates (as columns) """
  belief::B
  _partial::L
  """ just an example point for local access to the point data type"""
  _u0::P
  infoPerCoord::Vector{Float64}
end
const MKD{M,B,L} = ManifoldKernelDensity{M, B, L}


