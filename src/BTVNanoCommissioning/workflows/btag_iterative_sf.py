import logging
import warnings
from functools import reduce
from operator import and_, or_
from typing import Iterable

import awkward as ak
import hist
import numpy as np
from coffea import processor

from BTVNanoCommissioning.helpers.func import dump_lumi, update
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.correction import (
    common_shifts,
    load_lumi,
    load_SF,
    weight_manager,
)
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    MET_filters,
    ele_mvatightid,
    jet_id,
    mu_idiso,
)

# logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# ignore warnings from coffea
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*Missing cross-reference.*"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*divide by zero encountered in divide.*",
)


def reduce_and(*conditions):
    return reduce(and_, conditions)


def reduce_or(*conditions):
    return reduce(or_, conditions)


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


def nano_object_overlap(toclean, cleanagainst, dr=0.4):
    """Get Overlap mask between two collections of particles.
    Check if cleanagainst objects are outside of a certain delta R
    of toclean objects.
    """
    return ak.all(toclean.metric_table(cleanagainst) > dr, axis=-1)


def broadcast_and_flatten(arr1, arr2):
    """Broadcast two arrays and flatten the result."""
    arr1, arr2 = ak.broadcast_arrays(arr1, arr2)
    return ak.flatten(arr1, axis=None), ak.flatten(arr2, axis=None)


# define object selection functions


def muon_selection(events, campaign: str):
    """Object selection for muons.

    :param events: event array
    :type events: NanoAOD array
    :param campaign: campaign name
    :type campaign: str
    :return: selected muons
    :rtype: NanoAOD array
    """
    logger.debug("Selecting muons")
    muons = events.Muon
    mask = (muons.pt > 15) & (mu_idiso(events, campaign))
    return muons[mask]


def electron_selection(events, campaign: str):
    """Object selection for electrons.
    :param events: event array
    :type events: NanoAOD array
    :param campaign: campaign name
    :type campaign: str
    :return: selected electrons
    :rtype: NanoAOD array
    """
    logger.debug("Selecting electrons")
    electrons = events.Electron
    mask = (electrons.pt > 15) & (ele_mvatightid(events, campaign))
    return electrons[mask]


def jet_selection(events, campaign: str):
    """Object selection for jets.

    :param events: event array
    :type events: NanoAOD array
    :param campaign: campaign name
    :type campaign: str
    :return: selected jets
    :rtype: NanoAOD array
    """
    logger.debug("Selecting jets")
    jets = events.Jet
    mask = jet_id(events, campaign, max_eta=2.5, min_pt=20.0)
    return jets[mask]


def z_diamond(jets, met, leptons, dilep_mass):
    mht_x = ak.sum(jets.x, axis=-1) + ak.sum(leptons.x, axis=-1)
    mht_y = ak.sum(jets.y, axis=-1) + ak.sum(leptons.y, axis=-1)
    mht = np.sqrt(mht_x**2 + mht_y**2)
    return (
        (dilep_mass < (65.5 + 3 * mht / 8))
        | (dilep_mass > (108.0 - mht / 4))
        | (dilep_mass < (79.0 - 3 * mht / 4))
        | (dilep_mass > (99.0 + mht / 2))
    )


# define histograms to be filled
CHANNELS = {"ee", "mumu", "emu"}  # , "incl"
# CHANNELS = {"incl"}  # all four channels to large for 60 GB of RAM
HISTOGRAM_AXES = {
    "flav_axis": hist.axis.IntCategory([0, 1, 4, 5, 6], name="flav", label="Flavor"),
    "sys_axis": hist.axis.StrCategory([], name="syst", growth=True),
    "dr_axis": hist.axis.Regular(20, 0, 8, name="dr", label="$\Delta$R"),
    "pt_axis": hist.axis.Regular(
        60, 0, 300, name="pt", label=r"$p_{\mathrm{T}}$ / GeV"
    ),
    "pt_btag_axis": hist.axis.Variable(
        [*range(0, 200, 10), 200, np.inf], name="pt", label=r"$p_{\mathrm{T}}$ / GeV"
    ),
    "mass_axis": hist.axis.Regular(50, 0, 300, name="mass", label="$m$ / GeV"),
    "eta_axis": hist.axis.Regular(25, -2.5, 2.5, name="eta", label="$\eta$"),
    "abs_eta_axis": hist.axis.Variable(
        [0.0, 0.8, 1.6, 2.5], name="eta", label="|$\eta$|"
    ),
    "phi_axis": hist.axis.Regular(30, -3, 3, name="phi", label="$\phi$"),
    "region_axis": hist.axis.StrCategory(["HF", "LF"], name="region", label="Region"),
    "channel_axis": hist.axis.StrCategory(CHANNELS, name="channel", label="Channel"),
    "btagDeepFlavB_axis": hist.axis.Regular(
        100,
        0,
        1,
        name="btagDeepFlavB",
        label="btagDeepFlavB",
    ),
    "btagPNetB_axis": hist.axis.Regular(
        100,
        0,
        1,
        name="btagPNetB",
        label="btagPNetB",
    ),
    "btagUParTAK4B_axis": hist.axis.Regular(
        100,
        0,
        1,
        name="btagUParTAK4B",
        label="btagUParTAK4B",
    ),
    "btagRobustParTAK4B_axis": hist.axis.Regular(
        100,
        0,
        1,
        name="btagRobustParTAK4B",
        label="btagRobustParTAK4B",
    ),
}


def define_histograms(
    particle_objects: dict[str, list], b_taggers: Iterable[str], channels: Iterable[str]
):
    logger.debug("defining histograms")
    histograms = {
        "dr_jets": hist.Hist(
            HISTOGRAM_AXES["sys_axis"],
            HISTOGRAM_AXES["dr_axis"],
            hist.storage.Weight(),
        ),
        "njet": hist.Hist(
            HISTOGRAM_AXES["sys_axis"],
            hist.axis.Integer(0, 10, name="njet", label="N-jets"),
            hist.storage.Weight(),
        ),
    }

    for obj, attrs in particle_objects.items():
        for attr in attrs:
            histograms[f"{obj}_{attr}"] = hist.Hist(
                HISTOGRAM_AXES["sys_axis"],
                HISTOGRAM_AXES[f"{attr}_axis"],
                hist.storage.Weight(),
            )

    for b_tagger in b_taggers:
        for channel in channels:
            histograms[f"{b_tagger}_{channel}"] = hist.Hist(
                HISTOGRAM_AXES["sys_axis"],
                HISTOGRAM_AXES["flav_axis"],
                HISTOGRAM_AXES["abs_eta_axis"],
                HISTOGRAM_AXES["pt_btag_axis"],
                HISTOGRAM_AXES["region_axis"],
                # HISTOGRAM_AXES["channel_axis"],
                hist.axis.IntCategory([0, 1], name="jet_index", label="Jet index"),
                HISTOGRAM_AXES[f"{b_tagger}_axis"],
                hist.storage.Weight(),
                label=b_tagger,
            )

    return histograms


def fill_histograms(
    histograms,
    pruned_events,
    flavor_and_regions: dict[str, ak.Array],
    b_taggers: Iterable[str],
    particle_objects: dict[str, list],
    channels: Iterable[str],
    weights,
    systematics: Iterable,
    isSyst: bool,
):
    logger.debug("filling histograms")
    for syst in systematics:
        if not isSyst and syst != "nominal":
            break

        weight = (
            weights.weight()
            if syst == "nominal" or syst not in list(weights.variations)
            else weights.weight(modifier=syst)
        )

        if syst == "nominal":
            # fill only nominal
            # kinematic distributions
            for obj, attrs in particle_objects.items():
                for attr in attrs:
                    # if # in obj, use the index to access the correct object
                    parts = obj.split("#", 1)  # maxsplit=1 ensures at most 2 parts
                    _obj = parts[0]
                    if len(parts) > 1:
                        index = int(parts[1])
                        attr_value = getattr(pruned_events[_obj][:, index], attr)
                    else:
                        attr_value = getattr(pruned_events[_obj], attr)

                    # flatten array if more than one dimension
                    weight_flat = weight
                    if attr_value.ndim > 1:
                        weight_flat, attr_value = broadcast_and_flatten(
                            weight, attr_value
                        )

                    histograms[f"{obj}_{attr}"].fill(
                        **{attr: attr_value, "weight": weight_flat, "syst": syst},
                    )

            # delta R between jets
            histograms["dr_jets"].fill(
                syst=syst,
                weight=weight,
                dr=pruned_events["SelJet"][:, 0].delta_r(pruned_events["SelJet"][:, 1]),
            )

            # number of jets
            histograms["njet"].fill(
                njet=pruned_events.njet,
                weight=weight,
                syst=syst,
            )

        logger.debug("filling b-tag discriminant histograms for syst %s", syst)
        for b_tagger in b_taggers:
            for jet_index in (1, 0):
                for region in ("HF", "LF"):
                    for channel in channels:
                        # get the mask for the current region and channel
                        region_channel_mask = flavor_and_regions[
                            f"{region}_{channel}_{b_tagger}_probe_jet{jet_index}"
                        ]
                        if not ak.any(region_channel_mask):
                            continue
                        jet = pruned_events.SelJet[:, jet_index]

                        # fill the histogram
                        histograms[f"{b_tagger}_{channel}"].fill(
                            syst=syst,
                            flav=flavor_and_regions[
                                f"flavor_{b_tagger}_probe_jet{jet_index}"
                            ][region_channel_mask],
                            eta=np.abs(jet.eta[region_channel_mask]),
                            pt=jet.pt[region_channel_mask],
                            region=region,
                            # channel=channel,
                            jet_index=jet_index,
                            **{
                                b_tagger: flavor_and_regions[
                                    f"btag_{b_tagger}_probe_jet{jet_index}"
                                ][region_channel_mask]
                            },
                            weight=weight[region_channel_mask],
                        )

    return histograms


#######################
### processor class ###
#######################


class BTagIterativeSFProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year: str,
        campaign: str,
        outdir: str = "",
        isSyst: bool = False,
        isArray: bool = False,
        noHist: bool = False,
        chunksize: int = 75000,
        channel_selector: str = "incl",
    ):
        self._year = year
        self._campaign = campaign
        self._outdir = outdir
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.chunksize = chunksize
        self._channel = channel_selector

        # lumi mask
        self.lumiMask = load_lumi(self._campaign)

        # Load corrections
        self.SF_map = load_SF(self._year, self._campaign)

        # btagger configuration
        if self._year == "2024":
            ParT_name = "btagUParTAK4B"
        else:
            ParT_name = "btagRobustParTAK4B"

        self.b_tagger_config = {
            "btagDeepFlavB": {
                "loose": 0.0614,
                "medium": 0.3196,
                "tight": 0.73,
            },
            # "btagPNetB": {
            #     "loose": 0.1,
            #     "medium": 0.5,
            #     "tight": 0.9,
            # },
            ParT_name: {
                "loose": 0.0897,
                "medium": 0.451,
                "tight": 0.8604,
            },
        }

        self.particle_objects = {
            # "SelJet#0": [
            #     "pt",
            #     "eta",
            #     "phi",
            #     "mass",
            # ],  # , *self.b_tagger_config.keys()],
            # "SelJet#1": [
            #     "pt",
            #     "eta",
            #     "phi",
            #     "mass",
            # ],  # , *self.b_tagger_config.keys()],
            "PuppiMET": ["pt", "phi"],
            "dilep": ["pt", "eta", "phi", "mass"],
            "SelMuon": ["pt", "eta", "phi"],
            # "SelElectron": ["pt", "eta", "phi"],
        }

    def process(self, events):
        events = missing_branch(events)
        logger.info("preparing shifts")
        shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        logger.info("processing shift %s", shift_name)
        dataset = events.metadata["dataset"]

        # get histogram dict
        output = {}
        # get information if MC or data
        isRealData = not hasattr(events, "genWeight")
        if shift_name is None:
            if isRealData:
                output["sumw"] = len(events)
            else:
                output["sumw"] = ak.sum(events.genWeight)

        ####################
        #    Selections    #
        ####################

        # lumi mask
        logger.debug("Applying luminosity mask")
        lumi_mask = np.ones(len(events), dtype="bool")
        if isRealData:
            lumi_mask = self.lumiMask(events.run, events.luminosityBlock)
        if shift_name is None:
            output = dump_lumi(events[lumi_mask], output)

        # trigger
        triggers = {
            "ee": [
                "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
                "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            ],
            "emu": [
                "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
                "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            ],
            "mumu": [
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            ],
        }

        logger.debug("Getting trigger masks")
        trigger_masks = {
            channel: HLT_helper(events, trigger_paths)
            for channel, trigger_paths in triggers.items()
        }

        ### object selection ###

        # muon selection
        good_muons = muon_selection(events, campaign=self._campaign)

        # electron selection
        good_electrons = electron_selection(events, campaign=self._campaign)

        # create lepton collection
        good_leptons = ak.with_name(
            ak.concatenate([good_muons, good_electrons], axis=1),
            "PtEtaPhiMCandidate",
        )

        # jet selection
        good_jets = jet_selection(events, campaign=self._campaign)
        good_jets = good_jets[nano_object_overlap(good_jets, good_leptons, dr=0.4)]

        # MET selection
        met_filter_mask = MET_filters(events, self._campaign)

        ### events selection ###
        logger.debug("Event selection")

        # lepton cuts
        # combine leptons into a dilepton pair
        lepton_combinations = ak.combinations(
            good_leptons[..., :2],
            n=2,
            replacement=False,
            axis=-1,
            fields=["l0", "l1"],
        )
        dilep = ak.firsts(lepton_combinations.l0 + lepton_combinations.l1)
        dilep_mass = ak.fill_none(dilep.mass, np.nan)

        zll_mass_min = dilep_mass >= 12.0
        zll_pt_cut = ak.fill_none(dilep.pt, np.nan) > 10.0
        # z mass window
        zll_mass_cut = np.abs(dilep_mass - 91.1876) <= 10.0
        zll_mass_veto = np.abs(dilep_mass - 91.1876) > 10.0

        # leptons should have opposite charge, dilep charge is zero
        ll_opp_charge = ak.fill_none(dilep.charge, np.nan) == 0

        # min dr between leptons
        dr_ll = min_dr(good_leptons[..., :2])
        dr_ll_cut = dr_ll > 0.2

        # leading lepton pt cuts
        leading_muon_pt_cut = ak.fill_none(ak.firsts(good_muons.pt), np.nan) > 25.0
        leading_electron_pt_cut = (
            ak.fill_none(ak.firsts(good_electrons.pt), np.nan) > 25.0
        )

        # MET
        met = events.PuppiMET
        met_pt_cut = met.pt > 30.0

        # jets
        two_jets = ak.num(good_jets) == 2

        # z_diamond
        z_diamond_mask = z_diamond(good_jets, met, good_leptons, dilep_mass)

        # cuts specific to regions
        lepton_region_cut_HF = reduce_and(met_pt_cut, zll_mass_veto)
        lepton_region_cut_LF = reduce_and(
            ~met_pt_cut, zll_mass_cut, zll_pt_cut, ~z_diamond_mask
        )

        # trigger
        trigger_ee = reduce_and(trigger_masks["ee"], leading_electron_pt_cut)
        trigger_mumu = reduce_and(trigger_masks["mumu"], leading_muon_pt_cut)
        trigger_emu = reduce_and(
            trigger_masks["emu"],
            reduce_or(leading_electron_pt_cut, leading_muon_pt_cut),
        )

        ### create channel masks ###
        ch_mumu = reduce_and(
            trigger_mumu,
            ak.num(good_muons) == 2,
            ak.num(good_electrons) == 0,
            zll_mass_min,
            ll_opp_charge,
            dr_ll_cut,
        )
        ch_ee = reduce_and(
            trigger_ee,
            ak.num(good_electrons) == 2,
            ak.num(good_muons) == 0,
            zll_mass_min,
            ll_opp_charge,
            dr_ll_cut,
        )
        ch_emu = reduce_and(
            trigger_emu,
            ak.num(good_electrons) == 1,
            ak.num(good_muons) == 1,
            zll_mass_min,
            ll_opp_charge,
            dr_ll_cut,
        )
        assert not ak.any(reduce_and(ch_mumu, ch_ee, ch_emu)), (
            "Event in multiple channels"
        )
        ch_incl = reduce_or(ch_ee, ch_mumu, ch_emu)
        channels = {
            "ee": ch_ee,
            "mumu": ch_mumu,
            "emu": ch_emu,
            # "incl": ch_incl,
        }
        if self._channel != "all":
            channels = {
                self._channel: channels[self._channel]
            }
        # assert set(CHANNELS) == set(channels.keys()), (
        #     "channels in histogram definition and channel masks do not match"
        # )

        common_masks = [
            met_filter_mask,
            lumi_mask,
            two_jets,
        ]

        ### define pruned_events ###
        event_level_mask = reduce_and(*common_masks, ch_incl)
        pruned_events = events[event_level_mask]

        # skip empty events
        if len(pruned_events) == 0:
            if self.isArray:
                array_writer(
                    self,
                    pruned_events,
                    events,
                    ["nominal"],
                    dataset,
                    isRealData,
                )
            return {dataset: output}

        pruned_events["SelJet"] = good_jets[event_level_mask][:, :2]
        # get number of jets distribution without cut on number of jets
        # does not work shape wise
        pruned_events["njet"] = ak.count(good_jets[event_level_mask].pt, axis=1)

        pruned_events["SelMuon"] = good_muons[event_level_mask][:, :2]
        pruned_events["SelElectron"] = good_electrons[event_level_mask][:, :2]
        pruned_events["dilep"] = dilep[event_level_mask]

        # ==============
        # tag and probe
        # ==============

        logger.debug("tag-and-probe")
        flavor_and_regions = {}
        for i_tag_jet, i_probe_jet in [(0, 1), (1, 0)]:
            # for swapping, use: [(0, 1), (1, 0)]
            for b_tagger, config in self.b_tagger_config.items():
                tag_jet = ak.firsts(good_jets[:, i_tag_jet : i_tag_jet + 1])
                probe_jet = ak.firsts(good_jets[:, i_probe_jet : i_probe_jet + 1])
                # get btag score of the tag jet
                b_score_tag_jet = ak.fill_none(tag_jet[b_tagger], np.nan)
                b_score_probe_jet = ak.fill_none(probe_jet[b_tagger], np.nan)

                # cuts on jets
                tag_jet_cut = {
                    "HF": b_score_tag_jet >= config["medium"],
                    "LF": b_score_tag_jet <= config["loose"],
                }

                # flavor of probe jet
                if isRealData:
                    flavor_probe_jet = ak.zeros_like(probe_jet.pt)
                else:
                    flavor_probe_jet = np.abs(
                        ak.fill_none(
                            probe_jet.hadronFlavour,
                            0,
                        )
                    )
                # b_flavor = flavor_probe_jet == 5
                # c_flavor = flavor_probe_jet == 4
                # light_flavor = (flavor_probe_jet < 4) | (flavor_probe_jet > 5)
                flavor_and_regions[f"flavor_{b_tagger}_probe_jet{i_probe_jet}"] = (
                    ak.values_astype(flavor_probe_jet[event_level_mask], int)
                )

                # btag
                flavor_and_regions[f"btag_{b_tagger}_probe_jet{i_probe_jet}"] = (
                    b_score_probe_jet[event_level_mask]
                )

                # cuts specific to regions and channels
                for channel, channel_mask in channels.items():
                # channel, channel_mask = self._channel, channels[self._channel]
                    flavor_and_regions[
                        f"HF_{channel}_{b_tagger}_probe_jet{i_probe_jet}"
                    ] = reduce_and(
                        *common_masks,
                        channel_mask,
                        tag_jet_cut["HF"],
                        lepton_region_cut_HF,
                    )[event_level_mask]
                    flavor_and_regions[
                        f"LF_{channel}_{b_tagger}_probe_jet{i_probe_jet}"
                    ] = reduce_and(
                        *common_masks,
                        channel_mask,
                        tag_jet_cut["LF"],
                        lepton_region_cut_LF,
                    )[event_level_mask]

        # ==============
        # weights
        # ==============
        logger.debug("setting up weight_manager")
        weights = weight_manager(pruned_events, self.SF_map, self.isSyst)
        if shift_name is None:
            # we only need JEC shifts
            systematics = ["nominal"]  # + list(weights.variations)
        else:
            systematics = [shift_name]

        if not isRealData:
            pruned_events["weight"] = weights.weight()
            for include_weight in weights.weightStatistics.keys():
                pruned_events[f"{include_weight}_weight"] = weights.partial_weight(
                    include=[include_weight]
                )

        # ===============
        # fill histograms
        # ===============
        if not self.noHist:
            histograms = define_histograms(
                particle_objects=self.particle_objects,
                b_taggers=self.b_tagger_config.keys(),
                channels=channels.keys(),
            )
            histograms = fill_histograms(
                histograms=histograms,
                pruned_events=pruned_events,
                flavor_and_regions=flavor_and_regions,
                b_taggers=self.b_tagger_config.keys(),
                particle_objects=self.particle_objects,
                channels=channels.keys(),
                weights=weights,
                systematics=systematics,
                isSyst=self.isSyst,
            )

            output.update(histograms)

        # ==============
        # write arrays
        # ==============
        if self.isArray:
            array_writer(
                processor_class=self,
                pruned_event=pruned_events,
                nano_event=events,
                weights=weights,
                systname=systematics[0],
                dataset=dataset,
                isRealData=isRealData,
                kinOnly=[],
                kins=[],
                othersData=["events"],
            )

        return {dataset: output}

    def postprocess(self, accumulator):
        # cross section needed for sf derivation
        from BTVNanoCommissioning.helpers.xsection import xsection

        accumulator["xsection"] = xsection
        return accumulator
