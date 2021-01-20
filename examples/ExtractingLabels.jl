# capturing MSGibbs' final label selection per sample

##

using ApproxManifoldProducts
# using KernelDensityEstimate

import ApproxManifoldProducts: KDE, _BiDictMap


##


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
  mdp = DensityProductElements(outArr, bw, outName, Ref(true), lblComb, (XXarr...,), (BWarr...,), (inNames...,), Channel{Pair{Int,Int}}(100))

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


##

# build _BiDictMap between variable products



##

X1 = kde!([1;2;3.0],[1.0;]);
X2 = kde!([0.5;1.5;2.5],[1.0;]);
X3 = kde!([4;5;6.0],[1.0;]);



##


dpe = _buildDensityProductElements([X1;X2;X3], inNames=[:X1, :X2, :X3], outName = :Y)




##

_recalcProductKernel(dpe, 1)

##

_recalcProductKernel(dpe, 2)

_recalcProductKernel(dpe, 3)


##

_listOutElementLabelSelections(dpe, 1)
_listOutElementLabelSelections(dpe, 2)
_listOutElementLabelSelections(dpe, 3)


##

using LightGraphs
using TensorCast


## new basic example

X_ = kde!(5 .+ randn(5), [1.0;]);
# Y_ = kde!(randn(5), [1.0;]);
Z_ = kde!(5 .+ randn(5), [1.0;]);


##

# Y_ = X_*Z_

dpeY = _buildDensityProductElements([X_; Z_], inNames=[:X, :Z], outName = :Y)


@cast pts[i,j] := dpeY.outElements[j][i]
Y_ = kde!(pts, dpe.outBW[1])


##

X = kde!(pts .- 5)
dpeX = _buildDensityProductElements([X;], inNames=[:Y;], outName = :X)

##

Z = kde!(pts .+ 5)
dpeZ = _buildDensityProductElements([Z;], inNames=[:Y;], outName = :Z)



##

struct _AMPDiGraph{G,B}
  dg::G
  bd::B
end

function buildDiGraphKernelProduct!(dpel::DensityProductElements,
                                    abg::_AMPDiGraph = _AMPDiGraph(DiGraph(), AMP._BiDictMap(sizehint=100))  )
  #

  dg = abg.dg
  _bd = abg.bd

  vari = dpel.inElemNames
  N = length(dpel.outElements)

  for s in vari
    # one vertex to represent the variable
    if !haskey(_bd, s)
      add_vertex!(dg)
      _bd[s] = nv(dg)
    end
    for i in 1:N
      # one vertex for each kernel in the vertex belief
      kers = Symbol(s,"_$i")
      if !haskey(_bd, kers)
        add_vertex!(dg)
        _bd[kers] = nv(dg)
        add_edge!(dg, _bd[kers], _bd[s])
      end
    end
  end


  ## add who is getting the product

  s = dpel.outElemName
  if !haskey(_bd, s)
    add_vertex!(dg)
    _bd[s] = nv(dg)
  
    for i in 1:length(dpel.outElements)
      # one vertex for each kernel in the vertex belief
      add_vertex!(dg)
      kers = Symbol(s,"_$i")
      _bd[kers] = nv(dg)
      add_edge!(dg, _bd[kers], _bd[s])
    end
  end

  ## go through product and add edges from product of proposals

  # dId incoming density ID
  # sId selected label ID
  # eId product element ID
  # kCmb vector of label combinations from incoming densities
  for (eId, kCmb) in enumerate(dpel.lblCombinations), (dId,sId) in enumerate(kCmb)
    # @show dId, eId, sId, kCmb
    # @show dpel.inElemNames
    from = Symbol(dpel.inElemNames[dId], "_$sId")
    to = Symbol(dpel.outElemName, "_$eId")
    # add if the edge does not yet exist
    if !(_bd[to] in outneighbors(dg, _bd[from]) )
      add_edge!(dg, _bd[from], _bd[to])
    end
  end

  return abg
end



##


abg = buildDiGraphKernelProduct!( dpeY )


##


abg = buildDiGraphKernelProduct!( dpeX, abg )
abg = buildDiGraphKernelProduct!( dpeZ, abg )



##

# lets see what happened
using GraphPlot, Compose, Cairo


##

pth = "/tmp/gplot.pdf"
gplot(abg.dg, nodelabel=(i->abg.bd[i]).(keys(abg.bd)) ) |> PDF(pth)
@async run(`evince $pth`)



##

using Gadfly
Gadfly.set_default_plot_size(25cm,20cm)


##


spy(adjacency_matrix(abg.dg))





## list all paths worth find gradients for


# from dependent to independent variable, `d/d ind  dep = dz/dx`
# used for TAF gradient calculations
function findPathsOfKernels(abg::_AMPDiGraph, dep::Symbol, ind::Symbol)
  #
  allpths = Vector{Vector{Symbol}}()

  # get number of independent variables
  allsyms = string.([abg.bd[n] for n in keys(abg.bd)])
  @show Ni = sum(occursin.("$(ind)_", allsyms) )
  @show Nj = sum(occursin.("$(dep)_", allsyms) )

  for i in 1:Ni, j in 1:Nj
    # get the path
    spth = dijkstra_shortest_paths(abg.dg, abg.bd[Symbol(dep,"_$i")])
    # trim to first hit on independent variable
    pth1 = enumerate_paths(spth, abg.bd[Symbol(ind, "_$j")]) .|> x->getindex(abg.bd, x)
    strp = string.(pth1)
    len = occursin.("$ind", strp)
    len_ = 0 < length(len) ? findfirst(len) : 0
    # trim start for first hit on dependent variable
    # pth1 = enumerate_paths(spth, abg.bd[Symbol(dep, "_$j")]) .|> x->getindex(abg.bd, x)
    stt = occursin.("$dep", strp)
    stt_ = 0 < length(stt) ? findlast(stt) : 0

    # assemble all paths
    stt_ < len_ ? push!(allpths, pth1[stt_:len_]) : nothing
  end

  # return all unique paths
  unique(allpths)
end

##


unqpths = findPathsOfKernels(abg, :X, :Z)


unqpths = findPathsOfKernels(abg, :Z, :X)


## get values and measurements associated with each selection

m = match(r"_[0-9]+", unqpths[1][1] |> string)
sId = parse(Int, split(m.match, '_')[end])
xval = dpeX.outElements[sId]


m = match(r"_[0-9]+", unqpths[1][end] |> string)
sId = parse(Int, split(m.match, '_')[end])
zval = dpeZ.outElements[sId]


## get measurements for each factor along path



unqpths[1]



## each path is a minimization problem that must be solved to get gradients





##



## ===============================================================================================


##

dpe.inElements[3][1]

# idea is not to `sequence` a change on own dpe, but rather inform other clique variable of a change 
# put on the other variable changes to those proposals 
_sequenceUpdateProposalElement(dpe, [4.0;], 3, 1)
# _updateProposalElement(dpe, 4.0, 3, 1)


##

# on the other variable now take the updated proposal element (incoming) and 
# find which outgoing Element needs to be updated
_takeOutElementFromSequence(dpe)



## test case

## the subgraph
# X--Y--Z

## find Y
# localProduct!(Y)
# this should list dpeY. with inProductLabel containing random X, Z labels
# update source factor label selections


## do single factor links last
# localProduct!(X)
# update source factor label selections

# localProduct!(Z)
# update source factor label selections


# 



#



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