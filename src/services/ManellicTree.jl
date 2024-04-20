


# function Base.getproperty(mt::ManellicTree{M,D,N},f::Symbol) where {M,D,N}
#   if f !== :kernel
#     getfield(mt, f)
#   else

#   end
# end


getPoints(
  mt::ManellicTree
) = view(mt.data, mt.permute)

getWeights(
  mt::ManellicTree
) = view(mt.weights, mt.permute)


# either leaf or tree kernel, if larger than N
leftIndex(
  ::ManellicTree, 
  krnIdx::Int=1
) = 2*krnIdx
rightIndex(
  ::ManellicTree, 
  krnIdx::Int
) = 2*krnIdx+1


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


_getleft(i::Integer) = 2*i
_getright(i::Integer) = 2*i + 1


function buildTree_Manellic!(
  mtree::ManellicTree{MT,D,N},
  index::Integer, # tree node root=1,left=2n,right=left+1
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

  lft = mid_idx <= low ? low : _getleft(index)
  rgt = high <= mid_idx+1 ? high : _getright(index)

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

  # set HyperEllipse at this level in tree
  # FIXME, replace with just the kernel choice, not hyper such and such needed?
  if index < N
    _knl = convert(eltype(mtree.tree_kernels), knl)
    mtree.tree_kernels[index] = _knl # HyperEllipse(knl.μ, knl.Σ.mat)
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

  _legacybw(s::AbstractMatrix) = s
  _legacybw(s::AbstractVector) = diagm(s)
  _legacybw(::Nothing) = CV

  lCV = _legacybw(kernel_bw)
  # lCV = if isnothing(kernel_bw)
  #   CV
  # else
  #   kernel_bw
  # end

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
    SizedVector{len,Set{Int}}(undef),
    MVector{len,Int}(undef),
    MVector{len,Int}(undef),
    Set{Int}()
  )

  #
  tosort_leaves = buildTree_Manellic!(
    mtree,
    1, # start at root
    1, # spanning all data
    len; # to end of data
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

Evaluate the belief density for a given Manellic tree.

DevNotes:
- Computational Geometry
  - use geometric computing for faster evaluation
- Dual tree evaluations
  - Holmes, M.P., Gray, A.G. and Isbell Jr, C.L., 2010. Fast kernel conditional density estimation: A dual-tree Monte Carlo approach. Computational statistics & data analysis, 54(7), pp.1707-1718.
- Fast kernels
- Parallel transport shortcuts?
"""
function evaluate(
  mt::ManellicTree{M,D,N,HL},
  p,
  # bw_scl::Real = 1,
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
    if !LOO || !isapprox(p, t) # FIXME change isapprox to on-manifold version
      # FIXME, is this assuming length(pts) and length(mt.leaf_kernels) are the same?
      ekr = mt.leaf_kernels[i]
      # TODO remember special handling for partials in the future
      oneval = mt.weights[i] * evaluate(mt.manifold, ekr, p) 
      oneval *= !LOO ? 1 : 1/(1-w[i])
      sumval += oneval
    end
  end
  
  return sumval
end


function expectedLogL(
  mt::ManellicTree{M,D,N},
  epts::AbstractVector,
  # bw_scl::Real = 1,
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
    # return mean(log.(eL))
  end
end
# pathelogical case return -Inf


entropy(
  mt::ManellicTree,
) = -expectedLogL(mt, getPoints(mt), true)

# leaveOneOutLogL(
#   mt::ManellicTree,
#   bw_scl::Real = 1,
# ) = entropy(mt, bw_scl)


(mt::ManellicTree)(
  evalpt::AbstractArray,
) = evaluate(mt, evalpt)


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

# ## Pseudo code

# X1 = fg[:x1] # Position{2}

# # prob density
# x1 = X1([12.3;0.7])
# x1 = X1([-8.8;0.7])

# X1_a = approxConvBelief(dfg, f1, :x1)
# X1_b = approxConvBelief(dfg, f2, :x1)

# _X1_ = X1_a*X1_b

## WIP Sequential Gibbs Product development


getKernelLeafAsTreeKer(mtr::ManellicTree{M,D,N,HL,HT}, idx::Int, permuted::Bool=true) where {M,D,N,HL,HT} = convert(HT,getKernelLeaf(mtr, idx % N, permuted))

function getKernelTree(
  mtr::ManellicTree{M,D,N}, 
  currIdx::Int, 
) where {M,D,N}
  # must return sorted given name signature "Tree"
  permuted = true
  
  if currIdx < N
    return mtr.tree_kernels[currIdx]
  else
    return getKernelLeafAsTreeKer(mtr, currIdx, permuted)
  end
end


# function getKernelsTreeLevelIdxs(
#   mtr::ManellicTree{M,D,N},
#   level::Int,
#   currIdx::Int = 1
# ) where {M,D,N}

#   # go to children if idx level too high
#   _idxlevel(idx) = floor(Int,log2(idx)) + 1
#   _exists(_i) = _i < N ? (_i in mtr._workaround_isdef_treekernel) : true # N from numPoints anyway # isdefined(mtr.leaf_kernels,_i)

#   #traverse tree
#   if level == _idxlevel(currIdx) && _exists(currIdx)
#     return Int[currIdx;]
#   end

#   @warn "getKernelsTreeLevelIdxs still has corner case issues when tree_kernel[N]" maxlog=10

#   # explore left and right
#   return vcat(
#     getKernelsTreeLevelIdxs(mtr,level,_getleft(currIdx)),
#     getKernelsTreeLevelIdxs(mtr,level,_getright(currIdx))
#   )
# end

# function getKernelsTreeLevel(
#   mtr::ManellicTree{M,D,N},
#   level::Int
# ) where {M,D,N}
#   # kernels of that level
#   levelIdxs = getKernelsTreeLevelIdxs(mtr,level)
#   getKernelTree.(Ref(mtr),levelIdxs)
# end



function sampleProductSeqGibbsLabel(
  M::AbstractManifold,
  proposals::AbstractVector,
  MC = 3,
)
  #
  # how many incoming proposals
  d = length(proposals)

  ## TODO sample at multiscale levels on the tree, starting at treeLevel=1

  # how many points per proposal at this depth level of the belief tree
  Ns = proposals .|> getPoints .|> length

  # TODO upgrade to multiscale
  # start with random selection of labels
  latest_labels = [rand(1:n) for n in Ns]
  
  # pick the next leave-out proposal
  for _ in MC, O in 1:d
    # select a label from the not-selected proposal densities
    sublabels = Tuple{Int,Int}[]
    for s in setdiff(1:d, O)
      slbl = latest_labels[s]
      push!(sublabels, (s,slbl))
    end
    # prop = proposals[s]

    # TODO upgrade to map and tuples
    components = map(sl->getKernelLeaf(proposals[sl[1]], sl[2], true), sublabels)
    
    # calc product of Gaussians from currently selected LOO-proposals
    newO = calcProductGaussians(M, [components...])
    
    # evaluate new sampling weights of points in out component
    # NOTE getPoints returns the sorted (permuted) list of data
    evat = getPoints(proposals[O]) # FIXME how should partials be handled here?
    smw = zeros(length(evat))
    # FIXME, use multipoint evaluation such as NN (not just one point at a time)
    for (i,ev) in enumerate(evat)
      smw[i] = evaluate(M, newO, ev)
      # δc = distanceMalahanobisCoordinates(M,newO,ev)
    end
    
    smw ./= sum(smw)

    # smw = evaluate(newO, )
    p = Categorical(smw)
    
    # update label-distribution of out-proposal from product of selected LOO-proposal components
    latest_labels[O] = rand(p)
  end

  # recursively sample a layer deeper for each selected label, or fix that sample if that label is a leaf kernel
  for (i,l) in enumerate(latest_labels)

  end

  #
  return latest_labels
end


function sampleProductSeqGibbsLabels(
  M::AbstractManifold,
  proposals::AbstractVector,
  MC = 3,
  N::Int = round(Int, mean(length.(getPoints.(proposals))))
)
  #
  d = length(proposals)
  posterior_labels = Vector{NTuple{d,Int}}(undef,N)

  for i in 1:N
    posterior_labels[i] = tuple(sampleProductSeqGibbsLabel(M,proposals,MC)...)
  end

  posterior_labels
end


function calcProductKernelLabels(
  M::AbstractManifold,
  proposals::AbstractVector,
  lbls::AbstractVector{<:NTuple}
)
  #

  post = []

  for lbs in lbls
    # FIXME different tree or leaf kernels would need different lists
    props = MvNormalKernel[]
    for (i,lb) in enumerate(lbs)
      # selection of labels was done against sorted list of particles, hence false
      push!(props, getKernelLeaf(proposals[i],lb,false)) 
    end
    push!(post,calcProductGaussians(M,props))
  end

  return post
end

##
