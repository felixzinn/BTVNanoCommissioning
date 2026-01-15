from functools import reduce
from operator import and_, or_

import awkward as ak
import numpy as np


def nano_object_overlap(toclean, cleanagainst, dr=0.4):
    """Get Overlap mask between two collections of particles.
    Check if cleanagainst objects are outside of a certain delta R
    of toclean objects.
    """
    return ak.all(toclean.metric_table(cleanagainst) > dr, axis=-1)


def reduce_and(*masks):
    return reduce(and_, masks)


def reduce_or(*masks):
    return reduce(or_, masks)


def fill_none(array, fill_value=np.nan):
    return ak.fill_none(array, fill_value)


def make_p4(obj):
    """Generate 4-vector from a particle object."""
    return ak.zip(
        {
            "pt": obj.pt,
            "eta": obj.eta,
            "phi": obj.phi,
            "mass": obj.mass,
        },
        with_name="PtEtaPhiMCandidate",
    )


def min_dr(particles):
    """Get minimum delta R between pairs of particles."""
    di_particles = ak.combinations(
        particles,
        n=2,
        replacement=False,
        axis=1,
        fields=["p0", "p1"],
    )
    return ak.min(
        make_p4(di_particles.p0).delta_r(make_p4(di_particles.p1)),
        axis=-1,
        mask_identity=False,
    )
