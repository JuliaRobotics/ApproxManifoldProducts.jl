



# """
#     $TYPEDEF

# Elliptical structure for use in a (Manellic) Ball Tree.
# """
# struct HyperEllipse{P <:AbstractArray,D,DD}
#   """ manifold point at which this ellipse is based """
#   point::P
#   """ Covariance of coords at either TBD this point or some other reference point? """
#   coord_cov::SMatrix{D,D,Float64,DD}
# end

# ManellicTree

# Short for Manifold Ellipse Metric Tree
# starting as a balanced tree, relax to unbalanced in future.
struct ManellicTree{M,K,V<:AbstractVector,N}
  manifold::M
  data::V
  permute::MVector{N,Int}
  leaf_kernels::MVector{N,K}
  tree_kernels::MVector{N,K}
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
  r_PP::AbstractVector{P};
  kernel = MvNormal,
  kernel_bw = nothing,
) where {P <: AbstractArray}
  #
  # do calculations around mean point on manifold, i.e. support Riemannian
  p = mean(M, r_PP)
  
  r_XXp = log.(Ref(M), Ref(p), r_PP)
  r_CCp = vee.(Ref(M), Ref(p), r_XXp)
  
  D = manifold_dimension(M)
  # FIXME, consider user provided bandwidth in estimating multisample covariance
  cv = if 1 < length(r_PP) 
    SMatrix{D,D,Float64}(Manifolds.cov(M, r_PP))
  else
    # TODO case with user defined bandwidth for faster tree construction
    SMatrix{D,D,Float64}(diagm(ones(D)))
    return r_CCp, BitVector((true,)), kernel(p, cv)
  end

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
  ax_CCp, mask, kernel(p, cv)
end


_getleft(i::Integer) = 2*i
_getright(i::Integer) = 2*1 + 1


function buildTree_Manellic!(
  mtree::ManellicTree,
  index::Integer,
  low::Integer, # bottom index of segment
  high::Integer; # top index of segment;
  kernel = MvNormal,
  kernel_bw = nothing,
  leaf_size = 1
)
  M = mtree.manifold
  # take a slice of data
  idc = low:high
  # according to current index permutation (i.e. sort data as you build the tree)
  ido = view(mtree.permute, idc)
  # split the slice of order-permuted data
  ax_CCp, mask, knl = splitPointsEigen(M, view(mtree.data, ido); kernel)

  # set HyperEllipse at this level in tree
  # FIXME, replace with just the kernel choice, not hyper such and such needed?
  N = length(mtree.tree_kernels)
  if index < N
    mtree.tree_kernels[index] = knl # HyperEllipse(knl.μ, knl.Σ.mat)
  else
    mtree.leaf_kernels[index-N+1] = knl
  end

  # sort the data as 'small' and 'big' elements either side of the eigen split
  sml = view(ido, mask)
  big = view(ido, xor.(mask, true))
  # reorder the slice portion of the permutation with new ordering
  ido .= SA[sml...; big...]

  npts = high - low + 1
  mid_idx = low + sum(mask)

  if leaf_size < npts
    # recursively call two branches of tree, left
    buildTree_Manellic!(mtree, _getleft(index), low, mid_idx-1; kernel, leaf_size)
    # and right subtree
    buildTree_Manellic!(mtree, _getright(index), mid_idx, high; kernel, leaf_size)
  end

  return mtree
end



function buildTree_Manellic!(
  M::AbstractManifold,
  r_PP::AbstractVector{P}; # vector of points referenced to the r_frame
  kernel = MvNormal,
  bw = nothing # TODO
) where {P <: AbstractArray}
  #
  len = length(r_PP)
  D = manifold_dimension(M)
  tknlT = kernel(
    r_PP[1],
    SMatrix{D,D,Float64,D*D}(diagm(ones(D)))
  ) |> typeof
  lknlT = kernel(
    r_PP[1],
    if isnothing(bw)
      SMatrix{D,D,Float64,D*D}(diagm(ones(D)))
    else
      bw
    end
  )

  #
  mtree = ManellicTree(
    M,
    r_PP,
    MVector{len,Int}(1:len),
    MVector{len,lknlT}(undef),
    MVector{len,tknlT}(undef),
    MVector{len,Int}(undef),
    MVector{len,Int}(undef)
  )

  #
  return buildTree_Manellic!(
    mtree,
    1, # start at root
    1, # spanning all data
    len; # to end of data
    kernel
  )
end