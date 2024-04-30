


# function Base.getproperty(mt::ManellicTree{M,D,N},f::Symbol) where {M,D,N}
#   if f !== :kernel
#     getfield(mt, f)
#   else

#   end
# end

Base.length(::ManellicTree{M,D,N}) where {M,D,N} = N


getPoints(
  mt::ManellicTree
) = view(mt.data, mt.permute)

getWeights(
  mt::ManellicTree
) = view(mt.weights, mt.permute)


# _getleft(i::Integer, N) = 2*i + (2*i < N ? 0 : 1)
# _getright(i::Integer, N) = _getleft(i,N) + 1


# either leaf or tree kernel, if larger than N
leftIndex(
  mt::ManellicTree, 
  krnIdx::Int=1
) = 2*krnIdx + (2*krnIdx < length(mt) ? 0 : 1)

rightIndex(
  mt::ManellicTree, 
  krnIdx::Int
) = leftIndex(mt,krnIdx) + 1


"""
    $SIGNATURES

Return leaf kernel associated with input data element `i` (i.e. `permuted=true`).
Else when set to `permuted=false` return the sorted leaf_kernel `i` (different from unsorted input data number).
"""
getKernelLeaf(
  mt::ManellicTree,
  i::Int,
  permuted::Bool = true
) = mt.leaf_kernels[permuted ? mt.permute[i] : i]



"""
    $SIGNATURES

Return leaf kernels as tree kernel types, using regular `[1..N]` indexing].

Notes:
- use `permute=true` (default) for sorted index retrieval.
"""
getKernelLeafAsTreeKer(
  mtr::ManellicTree{M,D,N,HL,HT}, 
  idx::Int, 
  permuted::Bool=false
) where {M,D,N,HL,HT} = convert(HT,getKernelLeaf(mtr, (idx-1) % N + 1, permuted))

"""
    $SIGNATURES

Return kernel from tree by binary tree index, and convert leaf kernels to tree kernel types if necessary.

See also: [`getKernelLeafAsTreeKer`](@ref)
"""
function getKernelTree(
  mtr::ManellicTree{M,D,N,HL,HT}, 
  currIdx::Int, 
  # must return sorted given name signature "Tree"
  permuted = false,
  cov_continuation::Bool = false,
) where {M,D,N,HL,HT}
  #


  # BinaryTree (BT) index goes from root=1 to largest leaf 2*N
  if currIdx < N
    # cov_continuation correction so that we may build trees with sensible convariance to bandwidth transition from root to leaf
    raw_ker = mtr.tree_kernels[currIdx]
    if cov_continuation
      # depth of this index
      ances_depth = floor(Int, log2(currIdx))
      # how many leaf offsp
      offsp_depth = log2(length(mtr.segments[currIdx]))
      # get approx continuous depth fraction of this index
      λ = (ances_depth) / (ances_depth + offsp_depth)
      # mean bandwidth of all leaf children
      leafIdxs = mtr.segments[currIdx] .|> s->findfirst(==(s),mtr.permute)
      leafIdxs .+= N
      bws = [cov(getKernelTree(mtr,lidx,false)) for lidx in leafIdxs]
      # FIXME is a parallel transport needed between different kernel covariances that each exist in different tangent spaces
      mean_bw = mean(bws) # FIXME upgrade to on-manifold mean
      # corrected cov varies from root (only Monte Carlo cov est) to leaves (only selected bandwdith)
      nC = (1-λ)*cov(raw_ker) + λ*mean_bw
      # return a new kernel with cov_continuation, of tree kernel type
      kernelType = getfield(ApproxManifoldProducts,HT.name.name)
      kernelType(mean(raw_ker), nC, mtr.weights[currIdx])
    else
      raw_ker
    end
  else
    getKernelLeafAsTreeKer(mtr, currIdx, permuted)
  end
end


function exists_BTLabel(
  mt::ManellicTree{M,D,N},
  idx::Int
) where {M,D,N}
  # check for existence in tree or leaves
  eset = if idx < N 
    mt._workaround_isdef_treekernel
  else
    mt._workaround_isdef_leafkernel
  end

  # return existence
  return idx in eset
end

function isLeaf_BTLabel(
  mt::ManellicTree{M,D,N},
  idx::Int
) where {M,D,N}
  if exists_BTLabel(mt, leftIndex(mt,idx))
    return false
  elseif exists_BTLabel(mt, rightIndex(mt,idx))
    # TODO likely not needed to check for right child existence
    return false
  else
    return true
  end
end


uniWT(mt::ManellicTree) = 1===length(union(diff(getWeights(mt))))

function uniBW(
  mt::ManellicTree{M,D,N}
) where{M,D,N}
  if 1 < length(mt.leaf_kernels)
    bw = cov(mt.leaf_kernels[1])
    for lk in view(mt.leaf_kernels, 2:N)
      if !isapprox(bw, cov(lk))
        return false
      end
    end
  end
  return true
end


function Base.show(
  io::IO, 
  mt::ManellicTree{M,D,N,TK}
) where {M,D,N,TK}
  printstyled(io, "ManellicTree{"; bold=true,color = :blue)
  println(io)
  printstyled(io, "  M  = ", M, color = :magenta)
  println(io)
  printstyled(io, "  D  = ", D, color = :magenta)
  println(io)
  printstyled(io, "  N  = ", N, color = :magenta)
  println(io)
  printstyled(io, "  TK = ", TK, color = :magenta)
  # println(io)
  # printstyled(io, "  HT = ", HT, color = :magenta)
  println(io)
  printstyled(io, "}", bold=true, color = :blue)
  println(io, "(")
  @assert N == length(mt.data) "show(::ManellicTree,) noticed a data size issue, expecting N$(N) == length(.data)$(length(mt.data))"
  if 0 < N
    println(io, "  .data[1:]   :  ", mt.data[1], " ... ",    mt.data[end])
    println(io, "  .weights[1:]:  ", mt.weights[1], " ... ", mt.weights[end])
    printstyled(io, "     (uniwt)  :   ", uniWT(mt); color=:light_black)
    println(io)
    print(io, "  .permute[1:]:  ")
    printstyled(io, mt.permute[1], " ... ", mt.permute[end]; color=:light_black)
    println(io)
    print(io, "  .tkernels[") # " __see below__"; color=:light_black)
    if 0 < N
      # printstyled(io, "  .tkernels[1] = "; color=:light_black)
      print(io,"1]:  ")
      printstyled(io,"::TK ";color=:magenta)
      printstyled(io, mt.tree_kernels[1]; color=:light_black)
      # print(io, "  ...,")
    else
      print(io,"]:   ")
      printstyled(io,"::TK ";color=:magenta)
      println(io)
    end
    printstyled(io, "     (depth)  :   ", floor(Int,log2(length(mt.tree_kernels))),"+1"; color=:light_black)
    println(io)
    printstyled(io, "     (blncd)  :   ", "true : _wip_";color=:light_black)
    println(io)
    print(io, "  .lkernels[")
    if 0 < N
      print(io,"1]:  ")
      # printstyled(io, "  .tkernels[1] = "; color=:light_black)
      printstyled(io, mt.leaf_kernels[1]; color=:light_black)
      # print(io, "  ...,")
      if 1 < N
        printstyled(io, "         [end]:  "; color=:light_black)
        printstyled(io, mt.leaf_kernels[end]; color=:light_black)
      end
    else
      print(io,"]:   ")
      println(io)
    end
    # printstyled(io, "{1..$N}"; color=:light_black)
    # println(io)
    uBW = uniBW(mt)
    printstyled(io, "     (unibw)  :   ", uBW; color=:light_black)
    println(io)
    if uBW
      printstyled(io, "         bw   :    ", round.(getBW(mt)[1][:]';digits=3); color=:light_black)
      println(io)
    end
  end
  println(io, ")")
  # TODO ad dmore stats: max depth, widest point, longest chain, max clique size, average nr children

  return nothing
end

Base.show(io::IO, ::MIME"text/plain", mt::ManellicTree) = show(io, mt)


# case for identical types not requiring any conversions
Base.convert(
  ::Type{MvNormalKernel{P,T,M,iM}},
  src::MvNormalKernel{P,T,M,iM},
) where {P,T,M,iM} = src

function Base.convert(
  ::Type{MvNormalKernel{P,T,M,iM}},
  src::MvNormalKernel,
) where {P,T,M,iM}
  #
  _matType(::Type{Distributions.PDMats.PDMat{_F,_M}}) where {_F,_M} = _M
  μ = P(src.μ)
  p = MvNormal(_matType(M)(cov(src.p)))
  sqrt_iΣ = iM(src.sqrt_iΣ)
  MvNormalKernel{P,T,M,iM}(μ, p, sqrt_iΣ, src.weight)
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


"""
    $SIGNATURES

Give vector of manifold points and split along largest covariance (i.e. major direction)

DeVNotes:
- FIXME: upgrade to Manopt version 
  - https://github.com/JuliaRobotics/ApproxManifoldProducts.jl/issues/277
- FIXME use manifold mean and cov calculation instead
"""
function splitPointsEigen(
  M::AbstractManifold,
  r_PP::AbstractVector{P},
  weights::AbstractVector{<:Real} = ones(length(r_PP)); # FIXME, make static vector unless large
  kernel = MvNormalKernel,
  kernel_bw = nothing,
) where {P <: AbstractArray}
  #
  len = length(r_PP)
  
  # important, covariance is calculated around mean of points, which enables log to avoid singularities
  # do calculations around mean point on manifold, i.e. support Riemannian
  p = mean(M, r_PP)
  
  r_XXp = log.(Ref(M), Ref(p), r_PP) # FIXME replace with on-manifold distance
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
    # NOTE, rebalancing reason: deadcenter of covariance is not halfway between points (unconfirmed)
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

  weight = sum(weights)

  # return rotated coordinates and split mask
  ax_CCp, mask, kernel(p, cv, weight)
end



function buildTree_Manellic!(
  mtree::ManellicTree{MT,D,N},
  index::Integer, # tree node root=1,left=2n+corr,right=left+1
  low::Integer,   # bottom index of segment
  high::Integer;  # top index of segment;
  kernel = MvNormal,
  kernel_bw = nothing,
  leaf_size = 1
) where {MT,D,N}
  #
  _legacybw(s::Nothing) = s
  _legacybw(s::AbstractMatrix) = s
  _legacybw(s::AbstractVector) = diagm(s)

  _kernel_bw = _legacybw(kernel_bw)

  # terminate recursive tree build when all necessary tree kernels have been built
  if N <= index
    return mtree
  end

  M = mtree.manifold
  # take a slice of data
  idc = low:high
  # according to current index permutation (i.e. sort data as you build the tree)
  ido = view(mtree.permute, idc)
  # split the slice of order-permuted data
  ax_CCp, mask, knl = splitPointsEigen(M, view(mtree.data, ido), view(mtree.weights, ido); kernel, kernel_bw=_kernel_bw)
  imask = xor.(mask, true)

  # sort the data as 'small' and 'big' elements either side of the eigen split
  big = view(ido, mask)
  sml = view(ido, imask)
  # inplace reorder the slice portion of mtree.permute towards accending
  ido .= SA[sml...; big...]

  npts = high - low + 1
  mid_idx = low + sum(imask) - 1

  # @info "BUILD" index low sum(mask) mid_idx high _getleft(index) _getright(index)

  lft = mid_idx <= low ? low : leftIndex(mtree, index)
  rgt = high <= mid_idx+1 ? high : rightIndex(mtree, index)

  if leaf_size < npts
    if lft != low # mid_idx
      # recursively call two branches of tree, left
      buildTree_Manellic!(mtree, lft, low, mid_idx; kernel, kernel_bw=_kernel_bw, leaf_size)
    end
    if rgt != high
      # and right subtree
      buildTree_Manellic!(mtree, rgt, mid_idx+1, high; kernel, kernel_bw=_kernel_bw, leaf_size)
    end
  end

  if index < N
    _knl = convert(eltype(mtree.tree_kernels), knl)
    # FIXME use consolidate getKernelTree instead
    mtree.tree_kernels[index] = _knl 
    push!(mtree._workaround_isdef_treekernel, index)
    mtree.segments[index] = Set(ido)
  end

  return mtree
end


"""
    $SIGNATURES

Notes:
- Bandwidths for leaves (i.e. `kernel_bw`) must be passed in as covariances when `MvNormalKernel`.

DevNotes:
- Design Decision 24Q1, Manellic.MvNormalKernel bandwidth defs should ALWAYS ONLY BE covariances, because
  - Vision state is multiple bandwidth kernels including off diagonals in both tree or leaf kernels
  - Hybrid parametric to leafs convariance continuity
  - https://github.com/JuliaStats/Distributions.jl/blob/a9b0e3c99c8dda367f69b2dbbdfa4530c810e3d7/src/multivariate/mvnormal.jl#L220-L224
"""
function buildTree_Manellic!(
  M::AbstractManifold,
  r_PP::AbstractVector{P}; # vector of points referenced to the r_frame
  N = length(r_PP),
  weights::AbstractVector{<:Real} = ones(N).*(1/N),
  kernel = MvNormalKernel,
  kernel_bw = nothing, # TODO
) where {P <: AbstractArray}
  #
  D = manifold_dimension(M)
  CV = SMatrix{D,D,Float64,D*D}(diagm(ones(D))) 
  tknlT = kernel(
    r_PP[1],
    CV
  ) |> typeof

  _legacybw(s::AbstractMatrix) = s
  _legacybw(s::AbstractVector) = diagm(s)
  _legacybw(::Nothing) = CV

  lCV = _legacybw(kernel_bw)

  lknlT = kernel(
    r_PP[1],
    lCV
  ) |> typeof

  # kernel scale

  # leaf kernels
  lkern = SizedVector{N,lknlT}(undef)
  _workaround_isdef_leafkernel = Set{Int}()
  for i in 1:N
    lkern[i] = kernel(r_PP[i], lCV)
    push!(_workaround_isdef_leafkernel, i + N)
  end
  
  mtree = ManellicTree(
    M,
    r_PP,
    MVector{N,Float64}(weights),
    MVector{N,Int}(1:N),
    lkern, # MVector{N,lknlT}(undef),
    SizedVector{N,tknlT}(undef),
    # SizedVector{N,tknlT}(undef),
    SizedVector{N,Set{Int}}(undef),
    MVector{N,Int}(undef),
    MVector{N,Int}(undef),
    Set{Int}(),
    _workaround_isdef_leafkernel
  )

  #
  tosort_leaves = buildTree_Manellic!(
    mtree,
    1, # start at root
    1, # spanning all data
    N; # to end of data
    kernel,
    kernel_bw
  )

  # manual reset leaves in the order discovered
  permute!(tosort_leaves.leaf_kernels, tosort_leaves.permute)
    # dupl = deepcopy(tosort_leaves.leaf_kernels)
    # for (k,i) in enumerate(tosort_leaves.permute)
    #   tosort_leaves[i] = dupl.leaf_kernels[k]
    # end

  return tosort_leaves
end


"""
    $SIGNATURES
    
For Manellic tree parent kernels, what is the 'smallest' and 'biggest' covariance.

Notes:
- Thought about `det` for covariance volume but long access of pancake (smaller volume) is not minimum compared to circular covariance. 
"""
function getBandwidthSearchBounds(
  mtree::ManellicTree
)
  upper = cov(mtree.tree_kernels[1])

  #FIXME isdefined does not work as expected for mtree.tree_kernels, so using length-1 for now
  # this will break if number of points is not a power of 2. 
  kernels_diag = map(1:length(mtree.tree_kernels)-1) do i
    # FIXME use cosnolidated getKernelTree instead
    diag(cov(mtree.tree_kernels[i]))
  end
  lower_diag = minimum(reduce(hcat, kernels_diag), dims=2)

  # floors make us feel safe, but hurt when faceplanting
  lower_diag = maximum(
    hcat(
      lower_diag, 
      1e-8*ones(length(lower_diag))
    ), 
    dims=2
  )[:]

  # Give back lower as diagonal only covariance matrix
  lower = diagm(lower_diag)

  return lower, upper
end


"""
    $SIGNATURES

Evaluate the belief density for a given Manellic tree.

DevNotes:
- Computational Geometry
  - use geometric computing for faster evaluation
- Dual tree evaluations
  - Holmes, M.P., Gray, A.G. and Isbell Jr, C.L., 2010. Fast kernel conditional density estimation: A dual-tree Monte Carlo approach. Computational statistics & data analysis, 54(7), pp.1707-1718.
  - Curtin, R., March, W., Ram, P., Anderson, D., Gray, A. and Isbell, C., 2013, May. Tree-independent dual-tree algorithms. In International Conference on Machine Learning (pp. 1435-1443). PMLR.
- Fast kernels
- Parallel transport shortcuts?
"""
function evaluate(
  mt::ManellicTree{M,D,N,HL},
  pt,
  LOO::Bool = false,
) where {M,D,N,HL}
  # force function barrier, just to be sure dyndispatch is limited
  _F() = getfield(ApproxManifoldProducts,HL.name.name)
  _F_ = _F() 

  pts = getPoints(mt)
  w = getWeights(mt)

  sumval = 0.0
  # FIXME, brute force for loop
  for (i,t) in enumerate(pts)
    # FIXME change isapprox to on-manifold version
    if !LOO || !isapprox(pt, t) 
      # FIXME, is this assuming length(pts) and length(mt.leaf_kernels) are the same?
      # FIXME use consolidated getKernelLeaf instead
      ekr = mt.leaf_kernels[i]
      # TODO remember special handling for partials in the future
      oneval = mt.weights[i] * evaluate(mt.manifold, ekr, pt) 
      oneval *= !LOO ? 1 : 1/(1-w[i])
      sumval += oneval
    end
  end
  
  return sumval
end


"""
    $SIGNATURES

Return vector of weights of evaluated proposal label points against density.

DevNotes:
- TODO should evat points be of equal weights?  If multiscale sampling goes down unbalanced trees?
- FIXME how should partials be handled here? 
- FIXME, use multipoint evaluation such as NN (not just one point at a time)
"""
function evaluateDensityAtPoints(
  M::AbstractManifold,
  density,
  eval_at_points,
  normalize::Bool = true
)
  # evaluate new sampling weights of points in out component
  
  # vector for storing resulting weights
  smw = zeros(length(eval_at_points))
  for (i,ev) in enumerate(eval_at_points)
    # single kernel evaluation
    smw[i] = evaluate(M, density, ev) 
    # δc = distanceMalahanobisCoordinates(M,tmp_product,ev)
  end
  
  if normalize
    smw ./= sum(smw)
  end

  # return weights
  return smw
end



function expectedLogL(
  mt::ManellicTree{M,D,N},
  epts::AbstractVector,
  LOO::Bool = false
) where {M,D,N}
  T = Float64
  # TODO really slow brute force evaluation
  eL = MVector{length(epts),T}(undef)
  for (i,p) in enumerate(epts)
    # LOO skip for leave-one-out
    eL[i] = evaluate(mt, p, LOO)
  end
  # set numerical tolerance floor
  zrs = findall(isapprox.(0,eL))
  # nominal case with usable evaluation points
  eL[zrs] .= 1.0

  # weight and return within numerical reach
  w = getWeights(mt)
  if any(0 .!= w[zrs])
    -Inf
  else
    w'*(log.(eL))
    # return mean(log.(eL)) #?
  end
end


entropy(
  mt::ManellicTree,
) = -expectedLogL(mt, getPoints(mt), true)


(mt::ManellicTree)(
  evalpt::AbstractArray,
) = evaluate(mt, evalpt)



"""
    $SIGNATURES

Calculate one product of proposal kernels, as defined  BTLabels.
"""
function calcProductKernelBTLabels(
  M::AbstractManifold,
  proposals::AbstractVector,
  labels_sampled,
  LOOidx::Union{Int, Nothing} = nothing,
  gibbsSeq = 1:length(proposals);
  permute::Bool = true # true because signature is BTLabels
)
  # select a density label from the other proposals
  prop_and_label = Tuple{Int,Int}[]
  for s in setdiff(gibbsSeq, isnothing(LOOidx) ? Int[] : Int[LOOidx;])
    # tuple of which leave-one-out-proposal and its new latest label selection
    push!(prop_and_label, (s, labels_sampled[s]))
  end
  # get raw kernels from tree, also as tree_kernel type
  # NOTE DO COVARIANCE CONTINUATION CORRECTION FOR DEPTH OF TREE KERNELS
  components = map(pr_lb->getKernelTree(proposals[pr_lb[1]], pr_lb[2], permute, true), prop_and_label)

  # TODO upgrade to tuples
  return calcProductGaussians(M, [components...])
end


function calcProductKernelsBTLabels(
  M::AbstractManifold,
  proposals::AbstractVector,
  N_lbl_sets::AbstractVector{<:NTuple},
  permute::Bool = true # true because signature is BTLabels
)
  #
  T = typeof(getKernelTree(proposals[1],1))
  post = Vector{T}(undef, length(N_lbl_sets))

  for (i,lbs) in enumerate(N_lbl_sets)
    post[i] = calcProductKernelBTLabels(M, proposals, lbs; permute)
  end

  return post
end


function generateLabelPoolRecursive(
  proposals::AbstractVector{<:ManellicTree},
  labels_sampled::AbstractVector{<:Integer}
)
  # NOTE at top of tree, selections will be [1,1]
  child_label_pools = Vector{Vector{Int}}()

  # Are all selected labels leaves?
  all_leaves = true
  for _ in 1:length(proposals)
    push!(child_label_pools, Vector{Int}())
  end  
  for (o,idx) in enumerate(labels_sampled)
    isleaf = true
    # add interval of left and right children for next scale label sampling
    if exists_BTLabel(proposals[o], leftIndex(proposals[o], idx))
      push!(child_label_pools[o], leftIndex(proposals[o], idx))
      isleaf = false
    end
    if exists_BTLabel(proposals[o], rightIndex(proposals[o], idx))
      push!(child_label_pools[o], rightIndex(proposals[o], idx))
      isleaf = false
    end
    all_leaves &= isleaf
    if isleaf
      push!(child_label_pools[o], idx)
    end
  end

  return child_label_pools, all_leaves
end


"""
    $SIGNATURES

Notes:
- Advise, 2<=MC to ensure multiscale works during decent transitions (TBD obsolete requirement)
- To force sequential Gibbs on leaves only, use:
  `label_pools = [[(length(getPoints(prop))+1):(2*length(getPoints(prop)));] for prop in proposals]`
- Taken from: Sudderth, E.B., Ihler, A.T., Isard, M., Freeman, W.T. and Willsky, A.S., 2010. 
  Nonparametric belief propagation. Communications of the ACM, 53(10), pp.95-103.
"""
function sampleProductSeqGibbsBTLabel(
  M::AbstractManifold,
  proposals::AbstractVector{<:ManellicTree},
  MC = 3,
  # pool of sampleable labels
  label_pools::Vector{Vector{Int}}= [[1:1;] for _ in proposals],
  labels_sampled::Vector{Int} = ones(Int, length(proposals));
  # multiscale_parents = nothing;
  MAX_RECURSE_DEPTH::Int = 24, # 2^24 is so deep
)
  #
  # how many incoming proposals
  d = length(proposals)
  gibbsSeq = 1:d

  # pick the next leave-out proposal
  for _ in 1:MC, O in gibbsSeq
    # on first pass labels_sampled come from parent-recursive as part of multi-scale (i.e. pre-homotopy) operations
    # calc product of Gaussians from currently selected \LOO-proposals
    tmp_product = calcProductKernelBTLabels(M, proposals, labels_sampled, O, gibbsSeq; permute=false)

    # evaluate new weights for set of points from LOO proposal means
    eval_at_points = [mean(getKernelTree(proposals[O], i, false)) for i in label_pools[O]]
    smw = evaluateDensityAtPoints(M, tmp_product, eval_at_points, true) # TBD: smw = evaluate(tmp_product, )

    # update label-distribution of out-proposal from product of selected LOO-proposal components
    p = Categorical(smw)    
    labels_sampled[O] = label_pools[O][rand(p)]
  end
  
  # construct new label pool for children in multiscale
  child_label_pools, all_leaves = generateLabelPoolRecursive(proposals, labels_sampled)

  # recursively call sampling down the multiscale tree ("pyramid") -- aka homotopy
  # limit recursion to MAX_RECURSE_DEPTH
  if 0<MAX_RECURSE_DEPTH && !all_leaves
    # @info "Recurse down manellic tree for multiscale product"
    labels_sampled_copy = deepcopy(labels_sampled)
    labels_sampled = sampleProductSeqGibbsBTLabel(
      M, 
      proposals, 
      MC, 
      child_label_pools, 
      labels_sampled_copy; 
      MAX_RECURSE_DEPTH=MAX_RECURSE_DEPTH-1
    )

    # TODO, [circa 2006, Rudoy & Wolfe] detailed balance (Hastings) by rejecting a multiscale decent given simulated or parallel tempering
    # recursive call of sampleProductSeqGibbsBTLabel but with same parameters as this function invokation, aka reject the decend
  end

  #
  return labels_sampled
end


function sampleProductSeqGibbsBTLabels(
  M::AbstractManifold,
  proposals::AbstractVector,
  MC = 3,
  N::Int = round(Int, mean(length.(getPoints.(proposals)))), # FIXME use getLength or length of proposal (not getPoints)
  label_pools=[[1:1;] for _ in proposals]
)
  #
  d = length(proposals)
  posterior_labels = Vector{NTuple{d,Int}}(undef,N)

  for i in 1:N
    posterior_labels[i] = tuple(sampleProductSeqGibbsBTLabel(M, proposals, MC, label_pools)...)
  end

  posterior_labels
end





##
