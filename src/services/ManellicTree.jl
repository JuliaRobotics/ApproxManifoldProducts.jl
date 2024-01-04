


# function Base.getproperty(mt::ManellicTree{M,D,N},f::Symbol) where {M,D,N}
#   if f !== :kernel
#     getfield(mt, f)
#   else

#   end
# end

function Base.show(io::IO, mt::ManellicTree{M,D,N,KT}) where {M,D,N,KT}
  printstyled(io, "ManellicTree{"; bold=true,color = :blue)
  println(io)
  printstyled(io, "  M  = ", M, color = :magenta)
  println(io)
  printstyled(io, "  D  = ", D, color = :magenta)
  println(io)
  printstyled(io, "  N  = ", N, color = :magenta)
  println(io)
  printstyled(io, "  KT = ", KT, color = :magenta)
  # println(io)
  # printstyled(io, "  HT = ", HT, color = :magenta)
  println(io)
  printstyled(io, "}", bold=true, color = :blue)
  println(io, "(")
  @assert N == length(mt.data) "show(::ManellicTree,) noticed a data size issue, expecting N$(N) == length(.data)$(length(mt.data))"
  if 0 < N
    println(io, "  .data[1:]:     ", mt.data[1], " ... ",    mt.data[end])
    println(io, "  .weights[1:]:  ", mt.weights[1], " ... ", mt.weights[end])
    println(io, "  .permute[1:]:  ", mt.permute[1], " ... ", mt.permute[end])
    printstyled(io, "  .tkernels[1]:  ", " __see below__"; color=:light_black)
    println(io)
    println(io, "  ...,")
  end
  println(io, ")")
  if 0 < N
    printstyled(io, "  .tkernels[1] = "; color=:light_black)
    println(io, mt.tree_kernels[1])
  end
  # TODO ad dmore stats: max depth, widest point, longest chain, max clique size, average nr children

  return nothing
end

Base.show(io::IO, ::MIME"text/plain", mt::ManellicTree) = show(io, mt)


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
  len = length(r_PP)
  
  # do calculations around mean point on manifold, i.e. support Riemannian
  p = mean(M, r_PP)
  
  r_XXp = log.(Ref(M), Ref(p), r_PP)
  r_CCp = vee.(Ref(M), Ref(p), r_XXp)
  
  D = manifold_dimension(M)
  ndia = ( (D-1) ÷ 2 + 1 ) * D
  # FIXME, consider user provided bandwidth in estimating multisample covariance
  cv = if ndia < len
    SMatrix{D,D,Float64}(Manifolds.cov(M, r_PP))
  elseif 1 < len <= ndia
    SMatrix{D,D,Float64}(diagm(diag(Manifolds.cov(M, r_PP))))
  else
    # TODO case with user defined bandwidth for faster tree construction
    bw = isnothing(kernel_bw) ? SMatrix{D,D,Float64}(diagm(eps(Float64)*ones(D))) : kernel_bw
    return r_CCp, BitVector(ntuple(i->true,Val(len))), kernel(p, bw)
  end
    # S = SymmetricPositiveDefinite(2)
    # @info "COV" cv LinearAlgebra.isposdef(cv) Manifolds.check_point(S,cv) len

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
_getright(i::Integer) = 2*i + 1


function buildTree_Manellic!(
  mtree::ManellicTree{MT,D,N},
  index::Integer,
  low::Integer, # bottom index of segment
  high::Integer; # top index of segment;
  kernel = MvNormal,
  kernel_bw = nothing,
  leaf_size = 1
) where {MT,D,N}
  #
  if N <= index
    return mtree
  end

  M = mtree.manifold
  # take a slice of data
  idc = low:high
  # according to current index permutation (i.e. sort data as you build the tree)
  ido = view(mtree.permute, idc)
  # split the slice of order-permuted data
  ax_CCp, mask, knl = splitPointsEigen(M, view(mtree.data, ido); kernel, kernel_bw)

  # sort the data as 'small' and 'big' elements either side of the eigen split
  sml = view(ido, mask)
  big = view(ido, xor.(mask, true))
  # reorder the slice portion of the permutation with new ordering
  ido .= SA[sml...; big...]

  npts = high - low + 1
  mid_idx = low + sum(mask) - 1

  # @info "BUILD" index low sum(mask) mid_idx high _getleft(index) _getright(index)

  lft = mid_idx <= low ? low : _getleft(index)
  rgt = high < mid_idx+1 ? high : _getright(index)

  if leaf_size < npts
    if lft != mid_idx
      # recursively call two branches of tree, left
      buildTree_Manellic!(mtree, lft, low, mid_idx; kernel, kernel_bw, leaf_size)
    end
    if rgt != high
      # and right subtree
      buildTree_Manellic!(mtree, rgt, mid_idx+1, high; kernel, kernel_bw, leaf_size)
    end
  end

  # set HyperEllipse at this level in tree
  # FIXME, replace with just the kernel choice, not hyper such and such needed?
  if index < N
    mtree.tree_kernels[index] = knl # HyperEllipse(knl.μ, knl.Σ.mat)
  end

  return mtree
end



function buildTree_Manellic!(
  M::AbstractManifold,
  r_PP::AbstractVector{P}; # vector of points referenced to the r_frame
  len = length(r_PP),
  weights::AbstractVector{<:Real} = ones(len).*(1/len),
  kernel = MvNormal,
  kernel_bw = nothing, # TODO
) where {P <: AbstractArray}
  #
  D = manifold_dimension(M)
  CV = SMatrix{D,D,Float64,D*D}(diagm(ones(D))) 
  tknlT = kernel(
    r_PP[1],
    CV
  ) |> typeof
  lCV = if isnothing(kernel_bw)
    CV
  else
    kernel_bw
  end
  lknlT = kernel(
    r_PP[1],
    lCV
  ) |> typeof

  # kernel scale

  # leaf kernels
  lkern = SizedVector{len,lknlT}(undef)
  for i in 1:len
    lkern[i] = kernel(r_PP[i], lCV)
  end
  
  
  mtree = ManellicTree(
    M,
    r_PP,
    MVector{len,Float64}(weights),
    MVector{len,Int}(1:len),
    lkern, # MVector{len,lknlT}(undef),
    SizedVector{len,tknlT}(undef),
    # SizedVector{len,tknlT}(undef),
    MVector{len,Int}(undef),
    MVector{len,Int}(undef)
  )

  #
  return buildTree_Manellic!(
    mtree,
    1, # start at root
    1, # spanning all data
    len; # to end of data
    kernel,
    kernel_bw
  )
end

# TODO use geometric computing for faster evaluation
# DevNotes:
#  - Computational Geometry
#  - Dual tree evaluations
#  - Fast kernels
#  - Parallel transport shortcuts?
function evaluate(
  mt::ManellicTree{M,D,N},
  p,
) where {M,D,N}

  dim = manifold_dimension(mt.manifold)
  sumval = 0.0
  for i in 1:N
    ekr = mt.leaf_kernels[i]
    nscl = 1/sqrt((2*pi)^dim * det(cov(ekr.p)))
    oneval = mt.weights[i] * nscl * ker(mt.manifold, ekr, p, 0.5, distanceMalahanobisSq)
    # @info "EVAL" i oneval
    sumval += oneval
  end
  
  return sumval
end


# ## Pseudo code

# X1 = fg[:x1] # Position{2}

# # prob density
# x1 = X1([12.3;0.7])
# x1 = X1([-8.8;0.7])

# X1_a = approxConvBelief(dfg, f1, :x1)
# X1_b = approxConvBelief(dfg, f2, :x1)

# _X1_ = X1_a*X1_b

#