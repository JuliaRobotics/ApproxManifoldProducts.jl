
export
    AbstractManifold,
    HomotopyDensity,
    ManifoldKernelDensity,
    *,
    isapprox,
    # API and util functions
    getKDEManifoldBandwidths,
    manifoldProduct,
    isPartial,
    calcProductGaussians

export getPoints, getBW, Ndim, Npts, getWeights
export getKernelLeaf, getKernelTree #, getKernelLeafAsTreeKer
# export getKDERange, getKDEMax, getKDEMean, getKDEfit
export sample, rand, resample, kld, minkld
export calcMean
export mean, cov, std, var
export getInfoPerCoord, getBandwidth
export marginal, antimarginal

export mmd!, mmd

# partial specific functions
export getManifold, getManifoldPartial, getPartial
export getPointRepr


export ConcentratedGaussianKernel

# TODO deprecate below
export HomotopyDensity_legacy