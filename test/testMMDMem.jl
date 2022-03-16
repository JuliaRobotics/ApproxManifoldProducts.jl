# test mmd memory usage

using ApproxManifoldProducts
using Manifolds


##
# @testset
##


M = TranslationGroup(1)

p1 = [randn(1) for i in 1:100]
p2 = [randn(1) for i in 1:100]

# P1 = manikde!(M, p1)
# P2 = manikde!(M, p2)

val = [0.0;]
mmd!(M, val, p1, p2)


##
# end
#