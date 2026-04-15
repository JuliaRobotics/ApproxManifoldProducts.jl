Major news relating to breaking changes in ApproxManifoldProducts.jl

## v0.11

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
