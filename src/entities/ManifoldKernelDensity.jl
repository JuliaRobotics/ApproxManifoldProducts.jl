

struct ManifoldKernelDensity{M <: MB.AbstractManifold{MB.ℝ}, B <: BallTreeDensity, L, P}
  manifold::M
  belief::B
  _partial::L
  """ just an example point for local access to the point data type"""
  _u0::P
end
const MKD{M,B,L} = ManifoldKernelDensity{M, B, L}


