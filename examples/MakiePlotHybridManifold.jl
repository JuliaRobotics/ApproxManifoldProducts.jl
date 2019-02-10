
using LinearAlgebra
using Random
using Distributions

# plot features require Gadfly, etc
using ApproxManifoldProducts

# more packages required
using TransformUtils


const LinAlg = LinearAlgebra
const KDE = KernelDensityEstimate
const TU = TransformUtils
const AMP = ApproxManifoldProducts


# temporarily required for experimental versions of on-manifold products
KDE.setForceEvalDirect!(true)


## create two densities






# two densities on a cylinder
p = manikde!(randn(2,100), (:Euclid, :Circular) )

pts2a = 3.0*randn(1,100).+5.0
pts2b = TransformUtils.wrapRad.(0.5*randn(1,100).+pi)
q = manikde!([pts2a;pts2b], (:Euclid, :Circular) )

# approximate the product between hybrid manifold densities
pq = manifoldProduct([p;q], (:Euclid, :Circular), Niter=2)
# getBW(pq)[:,1]



## plotting with Makie

using AbstractPlotting, FileIO
using Makie



N = 200
x = LinRange(-7.5, 7.5, N)
y = LinRange(-pi, pi, N)
zp = zeros(N,N)
zq = zeros(N,N)
zpq = zeros(N,N)

pt = zeros(2,1)
for i in 1:length(x), j in 1:length(y)
  pt[1,1] = x[i]
  pt[2,1] = y[j]
  zp[i,j] = p(pt,false, 1e-3, (+, addtheta), (-,difftheta))[1]
  zq[i,j] = q(pt,false, 1e-3, (+, addtheta), (-,difftheta))[1]
  zpq[i,j] = pq(pt,false, 1e-3, (+, addtheta), (-,difftheta))[1]
end



## Compound plot


scene = hbox(
        # contour(x, y, z, levels = 20, linewidth =3),
        contour(x, y, zpq, levels = 0, linewidth = 0, fillrange = true),
        contour(x, y, zq, levels = 0, linewidth = 0, fillrange = true),
        contour(x, y, zp, levels = 0, linewidth = 0, fillrange = true),
)



## simpler plot

scene = contour(x, y, zpq, levels = 0, linewidth = 0, fillrange = true)



# save("/tmp/test.png", scene)

## better wrap around




pts1a = 6.0*rand(1,100).-3.0
pts1b = TransformUtils.wrapRad.(0.3*randn(1,100).-0.6*pi)
p = manikde!([pts1a;pts1b], (:Euclid, :Circular) )


pts2a = 6.0*rand(1,100).-3.0
pts2b = TransformUtils.wrapRad.(0.3*randn(1,100).+0.6*pi)
q = manikde!([pts2a;pts2b], (:Euclid, :Circular) )



# approximate the product between hybrid manifold densities
pq = manifoldProduct([p;q], (:Euclid, :Circular), Niter=3)


N = 200
x = LinRange(-7.5, 7.5, N)
y = LinRange(-pi, pi, N)
zp = zeros(N,N)
zq = zeros(N,N)
zpq = zeros(N,N)

pt = zeros(2,1)
for i in 1:length(x), j in 1:length(y)
  pt[1,1] = x[i]
  pt[2,1] = y[j]
  zp[i,j] = p(pt,false, 1e-3, (+, addtheta), (-,difftheta))[1]
  zq[i,j] = q(pt,false, 1e-3, (+, addtheta), (-,difftheta))[1]
  zpq[i,j] = pq(pt,false, 1e-3, (+, addtheta), (-,difftheta))[1]
end


scene = hbox(
  # contour(x, y, z, levels = 20, linewidth =3),
  contour(x, y, zpq, levels = 0, linewidth = 0, fillrange = true),
  contour(x, y, zq, levels = 0, linewidth = 0, fillrange = true),
  contour(x, y, zp, levels = 0, linewidth = 0, fillrange = true),
)






## Save scene to file

# using AbstractPlotting, CairoMakie, FileIO
# AbstractPlotting.current_backend[] = CairoMakie.CairoBackend("/tmp/test.svg")
# open("/tmp/test.svg","w") do io
#   show(io, MIME"image/svg+xml"(), scene)
# end



#
