




struct HyperEllipse{P,D}
  """ manifold point at which this ellipse is based """
  point::P
  """ Covariance of coords at either TBD this point or some other reference point? """
  coord_cov::SMatrix{D,D,Float64,<:Integer}
end

# ManellicTree

# Short for Manifold Ellipse Metric Tree
# starting as a balanced tree, relax to unbalanced in future.
struct ManellicTree{M,N,P}
  manifold::M
  data::MVector{N,P}
  hyper_ellipse::MVector{N,<:HyperEllipse}
  left_idx::MVector{N,Int}
  right_idx::MVector{N,Int}
end

# covariance
function eigenCoords(
  f_CVp::AbstractMatrix
)
  function _decomp(
    evc::AbstractMatrix,
    evl::AbstractVector, 
    _toflip::Bool = det(evc) < 0
  )
    pidx = _toflip ? sortperm(evl; rev=true) : 1:length(evl)
    Q = evc[:,pidx]
    L = diagm( evl[pidx] )
    return Q, L, pidx
  end

  E = eigen(f_CVp)
  f_Q_ax, Λ, pidx = _decomp(E.vectors, E.values)
  # largest variance is on coord `dim = pidx[end]`
  # derotate cloud for easy split
  # swap points order left and right of split
  return f_Q_ax, Λ, pidx
end



# give vector of manifold points and split along largest covariance (i.e. major direction)
function splitPointsEigen(
  M::AbstractManifold,
  r_PP::AbstractVector{P}
) where {P <: AbstractArray}
  #
  # do calculations around mean point on manifold, i.e. support Riemannian
  p = mean(M, r_PP)
  cv = Manifolds.cov(M, r_PP)
  
  r_XXp = log.(Ref(M), Ref(p), r_PP)
  r_CCp = vee.(Ref(M), Ref(p), r_XXp)
  
  # expecting largest variation on coord dimension `pidx[end]`
  r_R_ax, Λ, pidx = eigenCoords(cv)
  ax_R_r = r_R_ax'

  # rotate coordinates
  ax_CCp = map(r_CCp) do r_Cp 
    ax_R_r*r_Cp
  end

  # this is a local test around base point p (not at global 0)
  mask = 0 .<= (ax_CCp .|> s->s[1])

  # TODO ALLOW BOTH BALANCED OR UNBALANCED MASK RETRIEVAL, STARTING WITH FORCED MASK BALANCING
  # rebalance if stochastic nearest estimates fall in wrong mask
  function _flipmask_minormax!(
    smlmask, 
    bigmask, 
    data; 
    argminmax::Function = argmin
  )
    N = length(smlmask)
    # move minimum mask points over to imask
    for k in 1:((sum(bigmask) - sum(smlmask)) ÷ 2)
      # keep flipping the minimum element from mask into imask set
      # note using first coord, ie.. x-axis as the split axis: `s->s[1]`
      mlis = (1:sum(bigmask))
      ami = argminmax(view(data, bigmask))
      idx = mlis[ ami ]
      # get idx from orginal list
      flipidx = view(1:N, bigmask)[idx]
      data[flipidx]
      bigmask[flipidx] = xor(bigmask[flipidx], true)
      smlmask[flipidx] = xor(smlmask[flipidx], true)
    end
    nothing
  end
  
  imask = xor.(mask,true)
  ax_CC1 = (s->s[1]).(ax_CCp)
  _flipmask_minormax!(imask, mask, ax_CC1; argminmax=argmin)
  _flipmask_minormax!(mask, imask, ax_CC1; argminmax=argmax)

  # return rotated coordinates and split mask
  ax_CCp, mask
end
