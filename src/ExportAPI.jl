
export isapprox
export ArrayPartition
export mean

export
    AbstractManifold,
    *,
    # API and util functions
    getKDEManifoldBandwidths,
    manifoldProduct,
    isPartial,
    calcProductGaussians,
    getStateKind

export getPoints, getBW, Ndim, Npts, getWeights
export getKernelLeaf, getKernelTree #, getKernelLeafAsTreeKer
# export getKDERange, getKDEMax, getKDEMean, getKDEfit
export sample, rand, resample, kld, minkld
export calcMean
export mean, cov, std, var
export getObservability, getBandwidth
export marginal, antimarginal

export mmd!, mmd

# partial specific functions
export getManifold, getManifoldPartial, getPartial
export getPointType
export getMajorsLength

export ConcentratedGaussianKernel

export HomotopyRepr, HomotopyReprLive, HomotopyReprDFG
export AbstractBinaryTreeDensity, BinaryTreeDensity, BinaryTruncFixedDepth
export getFormKind, getTopologyKind
export HomotopyDensityLive, HomotopyDensityDFG, HomotopyDensity

# TODO deprecate below
export HomotopyDensity_legacy

export ManifoldKernelDensity

