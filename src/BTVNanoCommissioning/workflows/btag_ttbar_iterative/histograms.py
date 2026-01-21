from typing import Iterable

import awkward as ak
import hist
import numpy as np
from hist import Hist, axis

from BTVNanoCommissioning.utils.histogramming.axes.common import axes
from BTVNanoCommissioning.workflows.btag_ttbar_iterative.utils import fill_none

axes["pt"].label = "$p_{T}$ / GeV"
axes["mass"].label = "mass / GeV"
axes["njet"] = axis.Integer(0, 10, name="njet", label="Number of jets")
axes["dR_jets"] = axis.Regular(
    16, 0, 8, name="dR_jets", label=r"$\Delta R$ between jets"
)
axes["btagDeepFlavB"] = axis.Regular(
    100, 0, 1, name="btagDeepFlavB", label="DeepFlavB Score"
)
axes["btagUParTAK4B"] = axis.Regular(
    100, 0, 1, name="btagUParTAK4B", label="UParTAK4B Score"
)
axes["btagRobustParTAK4B"] = axis.Regular(
    100, 0, 1, name="btagRobustParTAK4B", label="RobustParTAK4B Score"
)


def define_and_fill_histograms(
    histogram_config: dict[str, dict[str, tuple[str, ...]] | tuple[str, ...]],
    events,
    systematics: Iterable[str],
    is_syst: bool,
    weights,
) -> dict[str, Hist]:
    histograms = define_histograms(
        btagger_names=histogram_config["btaggers"],
        particle_properties=histogram_config["particle_properties"],
        event_variables=histogram_config["event_variables"],
    )
    if not is_syst:
        # if not fill weights, only fill nominal
        systematics = ["nominal"]
    return fill_histograms(
        histograms=histograms,
        events=events,
        systematics=systematics,
        weights=weights,
    )


def define_histograms(
    btagger_names: Iterable[str],
    particle_properties: dict[str, tuple[str]],
    event_variables: Iterable[str],
) -> dict[str, Hist]:
    abs_eta = axis.Variable([0.0, 0.8, 1.6, 2.5], name="eta", label="|$\eta$|")
    custom_pt = axis.Variable(
        [*range(0, 200, 10), 200, np.inf], name="pt", label=r"$p_{\mathrm{T}}$ / GeV"
    )
    region_axis = axis.StrCategory(["HF", "LF", "None"], name="region", label="Region")
    jet_index_axis = axis.IntCategory([0, 1], name="jet_index", label="Jet Index")

    # histograms
    histograms: dict[str, dict[str | tuple[str, str], Hist]] = {
        "particle_properties": {},
        "event_variables": {},
        "btag_scores": {},
    }
    histograms["particle_properties"] = {
        (obj, prop): Hist(
            # axes["syst"],
            axes[prop],
            storage=hist.storage.Weight(),
            label=f"{obj}_{prop}",
        )
        for obj, props in particle_properties.items()
        for prop in props
    }
    histograms["event_variables"] = {
        f"{var}": Hist(
            # axes["syst"],
            axes[var],
            storage=hist.storage.Weight(),
            label=var,
        )
        for var in event_variables
    }

    # b-tag score
    for btagger in btagger_names:
        histograms["btag_scores"][btagger] = Hist(
            axes["syst"],
            axes["flav"],
            abs_eta,
            custom_pt,
            region_axis,
            jet_index_axis,
            axes[btagger],
            storage=hist.storage.Weight(),
            label=btagger,
        )

    return histograms


def fill_histograms(
    histograms: dict[str, dict[str | tuple[str, str], Hist]],
    events,
    systematics: Iterable[str],
    weights,
) -> dict[str, Hist]:
    # loop over weight systematics
    for systematic in systematics:
        weight = (
            weights.weight()
            if (systematic == "nominal" or systematic not in list(weights.variations))
            else weights.weight(modifier=systematic)
        )
        weight = weight.reshape((-1, 1))  # needs to be 2d (events, 1)

        # NOTE: fill flattened is used to fill histograms from 2d arrays
        # so all arrays must be 2d (events, ...)
        # fill b-tag scores
        # ================
        # tag and probe
        # ================
        for i_probe_jet in (0, 1):
            probe_jet = events.SelJet[:, i_probe_jet]
            for btagger, histogram in histograms["btag_scores"].items():
                # warnings.warn("B-tag score filling not implemented yet.")
                b_score_probe = fill_none(probe_jet[btagger])

                # get flavor of probe jet, for data use 0
                hadron_flavor = getattr(
                    probe_jet, "hadronFlavour", ak.zeros_like(probe_jet.pt)
                )
                # everything unknown is considered light flavor
                flavor_probe = np.abs(fill_none(hadron_flavor, fill_value=0))
                flavor_probe = ak.values_astype(flavor_probe, int)

                histogram.fill(
                    syst=systematic,
                    flav=flavor_probe,
                    eta=np.abs(probe_jet.eta),
                    pt=probe_jet.pt,
                    region=events[f"region_{btagger}_{i_probe_jet}"],
                    jet_index=i_probe_jet,
                    **{
                        btagger: b_score_probe,
                    },
                    weight=weight,
                )

        if systematic == "nominal":
            # we do not need the shifts for other histograms
            # fill particle properties
            for (obj, prop), histogram in histograms["particle_properties"].items():
                val = getattr(events[obj], prop).to_numpy()
                if val.ndim < 2:
                    val = val.reshape((-1, 1))  # needs to be 2d (events, 1)
                histogram.fill_flattened(
                    # syst=systematic,
                    **{prop: val},
                    weight=weight,
                )

            # fill event variables
            for var, histogram in histograms["event_variables"].items():
                val = events[var].to_numpy()
                if val.ndim < 2:
                    val = val.reshape((-1, 1))  # needs to be 2d (events, 1)
                histogram.fill_flattened(
                    # syst=systematic,
                    **{var: val},
                    weight=weight,
                )

    return histograms
