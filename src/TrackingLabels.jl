


# when calculating a density product, there is a certain element selection of incoming elements
# wont yet allow too much hybridization in types but enough to get going 
struct DensityProductElements{D,Vo<:AbstractVector,B,Vi<:AbstractVector}
  outElements::Vo
  outBW::Vector{B}
  outElemName::Symbol
  outdatedBW::Base.RefValue{Bool}
  lblCombinations::Vector{Vector{Int64}}
  inElements::NTuple{D,Vi}
  inBWs::NTuple{D,<:AbstractVector{B}}
  inElemNames::NTuple{D,Symbol}
  inElemFactorNames::NTuple{D,Symbol}
  # Pair{inDens, elemIdx}
  _sequencedUpdates::Channel{Pair{Int,Int}}
end


function Base.show(io::IO, x::DensityProductElements)
  # println(io, )
  printstyled(io, "DensityProductElements:", color=:blue)
  println(io)
  println(io, "  Out Name:     ", x.outElemName )
  println(io, "   in names:    ", x.inElemNames)
  println(io, "   # densities: ", length(x.inElements))
  println(io, "  # dims:       ", length(x.inElements[1][1]))
  println(io, "  # out pts:    ", length(x.outElements))
  println(io, "  .outBW[1]~:   ", x.outBW[1] .|> x->round(x,digits=5) )
  println(io, "  .outdatedBW:  ", x.outdatedBW[] )
  # println(io)
end

Base.show(io::IO, ::MIME"text/plain", x::DensityProductElements) = show(io, x)


# assume `mdp.inElements` is already set and corroborates with `lbsChs`
function _setLabelCombinations!(mdp::DensityProductElements, 
                                lbsChs::Dict )
  #
  npts_ = length(lbsChs)
  ndens = length(lbsChs[1])

  resize!(mdp.outElements, npts_)
  resize!(mdp.lblCombinations, npts_)

  NptsD = length.(mdp.inElements)

  # how many points i in the product belief
  for i in 1:npts_
    # make sure memory allocations are correct
    resize!(mdp.lblCombinations[i], ndens)
    # how many densities to make the product
    for densid in 1:ndens
      # level in the kd tree
      level = length(lbsChs[i][densid])
      # get label selection for densid
      lbl = lbsChs[i][densid][level] - NptsD[densid]
      # which row in xarrTiles to mark
      mdp.lblCombinations[i][densid] = lbl
      # xarrTiles[densid][lbl,i] = 1
    end
  end
  
  nothing
end


# # can only do for Array, not view
# function _setProductElements!(mdp::DensityProductElements{D}, 
#                               prd::BallTreeDensity)
#   #
#   # also set the bandwidth
#   dim = Ndim(prd)
#   resize!(mdp.outBW, dim)

#   npts = Npts(prd)
#   for i in 1:
#     mdp.outBW[:,i] .= getBW(prd)[:,1] # fix for all elements
#   end
  
#   # set kernel center elements
#   resize!(mdp.outElements, )
#   for i in 1:Npts(prd)
#     resize!(mdp.outElements[i], dim)
#     mdp.outElements[i][:] .= getPoints(prd, i)
#   end
  
#   #
#   nothing
# end


##

"""
Likely new bug see KDE #70 
"""
function _buildDensityProductElements(XX::AbstractVector{B};
                                      outName::Symbol=:product,
                                      inNames::Union{<:AbstractVector{Symbol},NTuple{D,Symbol}}=[Symbol("belief$i") for i in 1:length(XX)],
                                      inFctNames::Union{<:AbstractVector{Symbol},NTuple{D,Symbol}}=[Symbol("factor$i") for i in 1:length(XX)],
                                      _glbs=KDE.makeEmptyGbGlb(recordChoosen=true),
                                      product::B = *(XX, glbs=_glbs, addEntropy=false ) ) where {B<: BallTreeDensity, D}
  #
  
  npts = Npts(product)
  ndim = Ndim(product)
  ndens = length(XX)
  
  # pts = getPoints(product)
  # @cast outArr[j][i] := pts[i,j]
  outArr = [getPoints(product,i) for i in 1:npts]
  bw = [getBW(product)[:,i] for i in 1:npts]
  lblComb = [zeros(Int,ndens) for i in 1:npts]
  
  # restructure points for incoming densities
  # indens = (XX...,)
  XXarr = [([getPoints(x, i) for i in 1:Npts(x)]) for x in XX]
  BWarr = [([getBW(x)[:,i] for i in 1:Npts(x)]) for x in XX]

  # build the object
  mdp = DensityProductElements(outArr, bw, outName, Ref(true), lblComb, (XXarr...,), (BWarr...,), (inNames...,), (inFctNames...,), Channel{Pair{Int,Int}}(100))

  # set the labels selections used for current product
  if length(XX) == 1
    resize!(mdp.lblCombinations, npts)
    for i in 1:npts
      resize!(mdp.lblCombinations[i],1)
      mdp.lblCombinations[i][1] = i
    end
  else
    _setLabelCombinations!(mdp, _glbs.labelsChoosen)
  end


  return mdp
end



function _recalcProductKernel(mdp::DensityProductElements, idx::Int)
  #
  dim = length(mdp.inElements[1][1])
  selections = mdp.lblCombinations[idx]

  # TODO still have to do per dimension...
  infor = zeros(dim)
  informn = zeros(dim)
  for (idd, lbid) in enumerate(selections)
    # dens = mdp.inElements[idd]
    # mn_i = getPoints(dens, lbid)
    mn_i = mdp.inElements[idd][lbid]
    recpcov = 1/mdp.inBWs[idd][lbid].^2
    infor[:] .+= recpcov[:] # NOTE currently only diagonals
    informn += recpcov.*mn_i
  end
  cov = 1 ./ infor
  mn = informn./infor
  return mn, cov
end

##


function _updateProposalElement(mdp::DensityProductElements, elemval, densid::Int, elemidx::Int)
  # won't work for non-vector elements
  mdp.inElements[densid][elemidx] .= elemval
  mdp.outdatedBW[] = false
  nothing
end

function _updateOutElement(dpe::DensityProductElements, elemval, elemidx::Int)
  # won't work for non-vector elements
  dpe.outElements[elemidx] .= elemval
  dpe.outdatedBW[] = false
  nothing
end


function _sequenceUpdateProposalElement(mdp::DensityProductElements, elemval, densid::Int, elemidx::Int)
  _updateProposalElement(mdp, elemval, densid, elemidx)
  newpair = densid=>elemidx
  put!(mdp._sequencedUpdates, newpair)
  nothing
end


function _sequenceUpdateOutElement(dpe::DensityProductElements, elemval, elemidx::Int)
  _updateOutElement(dpe, elemval, elemidx)
  @warn("WIP")
end


_listOutElementLabelSelections(mdp::DensityProductElements, idx::Int) = mdp.lblCombinations[idx]

function _findOutElementConnectedLabel(mdp::DensityProductElements, densidx, idx::Int)
  outelems = Int[]
  # check for all elements
  for outi in 1:length(mdp.outElements)
    # if there happens to be a hit (out product relies on an in label selection)
    if _listOutElementLabelSelections(mdp, idx)[densidx] == outi
      push!(outelems, outi)
    end
  end

  # return list of outElements that depend on one particular inbound kernel `densidx=>idx`
  return outelems
end

# if something in the queue effects an outElement, then update values.
function _takeOutElementFromSequence(mdp::DensityProductElements)
  if !isready(mdp._sequencedUpdates)
    return nothing
  end
  nextpair = take!(mdp._sequencedUpdates)
  
  outElemList = _findOutElementConnectedLabel(mdp, nextpair[1], nextpair[2])
  println("Connected label $outElemList")

  return nothing
end

