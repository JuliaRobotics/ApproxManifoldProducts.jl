Major news relating to breaking changes in ApproxManifoldProducts.jl

## v0.15 (26Q2)

 Better support for unbalanced trees, including right-child-only-nodes.
- Refactoring `partial::Union{Nothing, <:Tuple>}` for `HomotopyDensity` constructors, utils, and helpers.  This replaces legacy code that used partials as vector type. 
- `HomotopyDensity.representationkind` replaces multiple parameters and fields, including `.manifold`, `{partial}`, ... (breaking change)
- Use `HomotopyDensity_legacy(; partial)` instead of `HomotopyDensity{partial}(.;.)` (breaking change)
- Refactored `HomotopyDensity{...}` parameters. (breaking change)
- Rename `HomotopyDensity.observability` instead of `.infoPerCoord` (breaking change)
- Rename `HomotopyDensity.points` instead of `.data` (breaking change)
- Rename `HomotopyDensity.structure` instead of `.geometric_permute` (breaking change)
- Drop `HomotopyDensity.leaf_kernels`, switch to new approach with `.minors_detail` (breaking change)
- Adding `HomotopyDensity.majors_*` fields 
- Deprecation, use `getPointType` instead of `getPointRepr`. (breaking change)

## v0.14 (26Q2)

- Drop `ManifoldKernelDensity`, use `HomotopyDensity` instead.  Only limited deprecation compat remains. (breaking change)


## v0.13 (26Q2)

- Drop `TreeDensity` abstract, use only `HomotopyDensity`. (breaking change)
- Rename `ManellicTree` to `HomotopyDensity` and refactored all internal calls. (breaking change)
- Use accessors for legacy `ManifoldKernelDensity` objects, e.g. `getManifold, getPartial, getPointRepr` (breaking change).
- Strip down `ManifoldKernelDensity` to only a `.shim::HomotopyDensity` field, with large-ish compat footprint in Deprecated.jl (breaking change).


## v0.12 (26Q2)

- Drop BallTreeDensity and KernelDensityEstimate.jl entirely (breaking change).
- Drop export of AMP acronym (breaking change).
- Drop legacy code, files, and exports of `get2DLambda, get2DMu, get2DMuMin, resid2DLinear, solveresid2DLinear!, solveresid2DLinear` (breaking changes).
- Drop old deprecated code including `setPointsManiPartial!, productbelief, calcVariableCovarianceBasic` (breaking changes).
- Drop NLsolve dependency.
- Drop TransformUtils dependency.
- Replace internal usage of skew(SO3) with LieGroups.hat(SO3) instead (was never exported).
- Drop dependencies Require, Reexport.
- `ConcentratedGaussianKernel` replaces `MvNormalKernel`, `DensityKernel`.
- Drop `MvNormalKernel`, `DensityKernel` (breaking change).


## v0.11 (25Q4 - 26Q1)

- Towards integration with DFG v1.0, including halfway step towards HomotopyDensity refactoring.
- Full support for ManellicTree to finally close #41, including supporting partials as in earlier versions
- Penultimate step to entirely removing KernelDensityEstimate.jl dependency, which will be archived and is severly limited to Euclidean space, balanced tree products only, poor partial support, no outside access to multiscale sampling process, among many other limitations.

## v0.5 

- Upgrade to ManifoldsBase.jl v0.13.

## v0.4

- `ManifoldKernelDensity` is the primary density approximation method.
- `rand(::MKD,N)` now returns a `::Vector{P}` of points type `P`, not a matrix of coordinate columns.

## v0.3 

- Upgrade to ManifoldsBase.jl v0.11 with `AbstractManifold`.
- Start consolidating internal manifold definitions to Manifolds.jl definitions instead.

## v0.2

- Replace `ManifoldBelief` with `ManifoldKernelDensity`.
- Adopt `ManifoldsBase.Manifold{ℝ}` as default abstract, replace old self defined `Manifold`.
