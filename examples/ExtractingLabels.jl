# capturing MSGibbs' final label selection per sample

##

using ApproxManifoldProducts
# using KernelDensityEstimate

const KDE = KernelDensityEstimate


##

X1 = kde!([1;2;3.0],[1.0;]);
X2 = kde!([0.5;1.5;2.5],[1.0;]);
X3 = kde!([4;5;6.0],[1.0;]);

##


# wont yet allow too much hybridization in types but enough to get going 
struct MetricDensityProduct{D,Vo<:AbstractVector,B,Vi<:AbstractVector}
  outElements::Vo
  outBW::Vector{B}
  outdatedBW::Bool
  lblCombinations::Vector{Vector{Int64}}
  inElements::NTuple{D,Vi}
  inBWs::NTuple{D,<:AbstractVector{B}}
end


function Base.show(io::IO, x::MetricDensityProduct)
  # println(io, )
  printstyled(io, "MetricDensityProduct:", color=:blue)
  println(io)
  println(io, "  # dims:       ", length(x.inElements[1][1]))
  println(io, "  # out pts:    ", length(x.outElements))
  println(io, "  # densities:  ", length(x.inElements))
  # println(io)
end

Base.show(io::IO, ::MIME"text/plain", x::MetricDensityProduct) = show(io, x)


# assume `mdp.inElements` is already set and corroborates with `lbsChs`
function _setLabelCombinations!(mdp::MetricDensityProduct, 
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
# function _setProductElements!(mdp::MetricDensityProduct{D}, 
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

function _buildMetricDensityProduct(XX::AbstractVector{B};
                                    _glbs=KDE.makeEmptyGbGlb(recordChoosen=true),
                                    product::B = *(XX, glbs=_glbs, addEntropy=false ) ) where {B<: BallTreeDensity}
  #

  @show pts = getPoints(product)
  
  npts = Npts(product)
  ndim = Ndim(product)
  ndens = length(XX)
  
  @show outArr = [getPoints(product,i) for i in 1:npts]
  bw = [getBW(product)[:,i] for i in 1:npts]
  lblComb = [zeros(Int,ndens) for i in 1:npts]
  
  # restructure points for incoming densities
  # indens = (XX...,)
  XXarr = [([getPoints(x, i) for i in 1:Npts(x)]) for x in XX]
  BWarr = [([getBW(x)[:,i] for i in 1:Npts(x)]) for x in XX]

  # build the object
  mdp = MetricDensityProduct(outArr, bw, true, lblComb, (XXarr...,), (BWarr...,))

  # set the labels selections used for current product
  _setLabelCombinations!(mdp, _glbs.labelsChoosen)

  return mdp
end



function _recalcProductKernel(mdp::MetricDensityProduct, idx::Int)
  
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

# function _updateProposalDensity(mdp::MetricDensityProduct, elemval, densid, idx)

  


# end



##


mdp = _buildMetricDensityProduct([X1;X2;X3])




##

_recalcProductKernel(mdp, 1)


##

_recalcProductKernel(mdp, 2)

_recalcProductKernel(mdp, 3)


##



##

# gradient from kernel perturbation




## DEV CODE BELOW




glbs = KDE.makeEmptyGbGlb();
glbs.recordChoosen = true

p123 = *([X1;X2;X3], glbs=glbs, addEntropy=false );


## Analyze what happened


lc = glbs.labelsChoosen

Ndens = 3
XX = [X1;X2;X3]


for i in 1:3
  μ = zeros(Ndens)
  for densid in 1:Ndens
    level = length(lc[i][densid])
    μ[densid] = getPoints(XX[densid])[lc[i][densid][level] - 3]
  end

  Λμ = sum(μ)

  @show μ = 1/3*Λμ
end



##

#