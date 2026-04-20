
export
    # new local features
    MKD, # TODO deprecate
    AbstractManifold,
    HomotopyDensity,
    ManifoldKernelDensity,
    *,
    isapprox,
    # APi and util functions
    buildHybridManifoldCallbacks,
    getKDEManifoldBandwidths,
    manifoldProduct,
    manikde!,
    isPartial,
    calcProductGaussians

export getPoints, getBW, Ndim, Npts, getWeights
export getKDERange, getKDEMax, getKDEMean, getKDEfit
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