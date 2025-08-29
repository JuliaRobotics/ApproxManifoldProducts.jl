
# A few supported manifolds

export
    # general manifolds
    Euclidean,
    Circle,
    ℝ,
    Euclid,
    Euclid2,
    Euclid3,
    Euclid4,
    SE2_Manifold,
    SE3_Manifold,

    # special quirks that still need to be fixed
    Circle1
# Circular,

#

const _AMP_CIRCLE = Manifolds.Circle()

# this is just wrong and needs to be fixed
const Circle1 = Manifolds.Circle()

const Euclid = TranslationGroup(1)
const EuclideanManifold = Euclid

const Euclid2 = TranslationGroup(2)
const Euclid3 = TranslationGroup(3)
const Euclid4 = TranslationGroup(4)

# TODO if not easy simplification exists, then just deprecate this
const SE2_Manifold = SpecialEuclideanGroup(2; variant = :right)
const SE3_Manifold = SpecialEuclideanGroup(3; variant = :right)

#
