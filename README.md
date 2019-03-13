# ApproxManifoldProducts.jl

[![Build Status](https://travis-ci.org/JuliaRobotics/ApproxManifoldProducts.jl.svg?branch=master)](https://travis-ci.org/JuliaRobotics/ApproxManifoldProducts.jl)
[![codecov.io](https://codecov.io/github/JuliaRobotics/ApproxManifoldProducts.jl/coverage.svg?branch=master)](https://codecov.io/github/JuliaRobotics/ApproxManifoldProducts.jl?branch=master)


<!-- [![ApproxManifoldProducts](http://pkg.julialang.org/badges/ApproxManifoldProducts_0.7.svg)](http://pkg.julialang.org/?pkg=ApproxManifoldProducts&ver=0.7)
[![ApproxManifoldProducts](http://pkg.julialang.org/badges/ApproxManifoldProducts_1.0.svg)](http://pkg.julialang.org/?pkg=ApproxManifoldProducts&ver=1.0) -->

# Introduction

Approximate the product between infinite functional objects on a manifold -- i.e. belief products.  ApproxManifoldProducts (AMP) is intended for use with the [Caesar.jl](http://www.github.com/JuliaRobotics/Caesar.jl) framework.  AMP is intended for development and testing of various kernel embedding methods for approximating functional products.

See [Caesar documentation](http://www.juliarobotics.org/Caesar.jl/latest/) for more details.

# Installation

For Julia 0.7 and above press ']' in the Julia REPL:
```julia
pkg> add ApproxManifoldProducts
```

# Current Supported (Mixed) Manifolds

The following on-manifold function approximations:
- Euclidean (2DOF),
- S1/Circular (1DOF), or SO(2) equivalent.
- SE(2)
- S2 (not implemented yet)

Work in progress on so-called 'subgroup' mixed-manifolds, where DOFs are bunched together for particular manifolds:
- SO(3) / Quaternion,
- Plucker coordinates (SP(3)) for rigid transforms,
- SE(3) for rigid transforms.

> Implementation of multivariate methodology requires code to consider "double loops" that iterate over sub-groups, and then within each sub-group as required.  The supported list above can be implemented with "single loop" over all DOFs.

Any suggestions or issues welcome.
