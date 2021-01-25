# capturing MSGibbs' final label selection per sample

##

using ApproxManifoldProducts
# using KernelDensityEstimate



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

dpeY = _buildDensityProductElements([X_; Z_], outName = :Y, inNames=[:X, :Z], inFctNames=[:XYf1; :YZf1])


@cast pts[i,j] := dpeY.outElements[j][i]
Y_ = kde!(pts, dpeY.outBW[1])


##

X = kde!(pts .- 5)
dpeX = _buildDensityProductElements([X;], outName = :X, inNames=[:Y;], inFctNames=[:XYf1;])

##

Z = kde!(pts .+ 5)
dpeZ = _buildDensityProductElements([Z;], outName = :Z, inNames=[:Y;], inFctNames=[:YZf1;])



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

using IncrementalInference


##

varList = [:X, :Y, :Z]


whatvar = unqpths[1][1]

onei = outneighbors(abg.dg, abg.bd[whatvar]) .|> x->getindex(abg.bd, x)
intersect(varList, onei)


## get the measurement for each factor






## each path is a minimization problem that must be solved to get gradients


##


findFactorsBetweenFrom()



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