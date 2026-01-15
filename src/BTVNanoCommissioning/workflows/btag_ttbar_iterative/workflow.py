import os
from pathlib import Path

import awkward as ak
import numpy as np
from coffea import processor

from BTVNanoCommissioning.helpers.func import dump_lumi, update
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.correction import (
    common_shifts,
    load_lumi,
    load_SF,
    weight_manager,
)
from BTVNanoCommissioning.utils.selection import HLT_helper, btag_wp_dict
from BTVNanoCommissioning.workflows.btag_ttbar_iterative.histograms import (
    define_and_fill_histograms,
)
from BTVNanoCommissioning.workflows.btag_ttbar_iterative.selections import (
    electron_selection,
    jet_selection,
    met_selection,
    muon_selection,
)
from BTVNanoCommissioning.workflows.btag_ttbar_iterative.utils import (
    fill_none,
    min_dr,
    nano_object_overlap,
    reduce_and,
)


TAGGER_NAMES = {
    "btagDeepFlavB": "DeepFlav",
    "btagUParTAK4B": "UParTAK4",
    "btagRobustParTAK4B": "RobustParTAK4",
}


class BaseProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year: str,
        campaign: str,
        output_directory: os.PathLike | Path,
        isSyst: str,
        isArray: bool,
        noHist: bool,
        chunksize: int,
    ):
        # named exactly like this for corrections
        self._year = year
        self._campaign = campaign
        self.output_directory = output_directory
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.chunksize = chunksize

    def postprocess(self, accumulator):
        return accumulator


class BTagIterativeSFProcessor(BaseProcessor):
    def __init__(
        self,
        *args,
        channel: str = "mumu",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.channel = channel
        if self.channel not in ["mumu", "ee", "emu"]:
            raise ValueError(f"Channel {self.channel} not supported")

        # lumi mask
        self.lumi_mask = load_lumi(self._campaign)

        # correction SFs
        # needs to be named like that (correction.py)
        self.SF_map = load_SF(self._year, self._campaign)

        ParT_name: str = (
            "btagUParTAK4B" if self._year == "2024" else "btagRobustParTAK4B"
        )
        self.histogram_config: dict[
            str, dict[str, tuple[str, ...]] | tuple[str, ...]
        ] = {
            "btaggers": (ParT_name, "btagDeepFlavB"),
            "particle_properties": {
                "dilepton": ("pt", "eta", "phi", "mass"),
                "PuppiMET": ("pt", "phi"),
                "SelJet": ("pt", "eta", "phi", "btagDeepFlavB", ParT_name),
            },
            "event_variables": ("dR_jets", "njet"),
        }
        if self.channel == "mumu":
            self.histogram_config["particle_properties"].update(
                {"SelMuon": ("pt", "eta", "phi")}
            )
        elif self.channel == "ee":
            self.histogram_config["particle_properties"].update(
                {"SelElectron": ("pt", "eta", "phi")}
            )
        elif self.channel == "emu":
            self.histogram_config["particle_properties"].update(
                {
                    "SelMuon": ("pt", "eta", "phi"),
                    "SelElectron": ("pt", "eta", "phi"),
                }
            )

    # coffea stuff
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        events = missing_branch(events)
        vetoed_events, shifts = common_shifts(self, events)
        return processor.accumulate(
            self.process_shift(update(vetoed_events, events), shift_name)
            for events, shift_name in shifts
        )

    def process_shift(self, events, shift_name):
        print(f"Processing shift: {shift_name}")
        dataset = events.metadata["dataset"]
        is_real_data = not hasattr(events, "genWeight")
        shift_name = "nominal" if shift_name is None else shift_name
        is_nominal = shift_name == "nominal"

        output = {}
        n_events = len(events)
        if is_nominal:
            # gen weight, if data fill with ones
            output["sumw"] = ak.sum(getattr(events, "genWeight", np.ones(n_events)))

        # =====================
        # Event Selection
        # =====================

        # === lumi ===
        lumi_mask = np.full((n_events,), True)
        if is_real_data:
            lumi_mask = self.lumi_mask(events.run, events.luminosityBlock)
        # dump lumi for nominal case
        if is_nominal:
            output = dump_lumi(events[lumi_mask], output)

        # === trigger ===
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

        # === object selection ===
        # muon selection
        good_muons = muon_selection(events, campaign=self._campaign)
        # electron selection
        good_electrons = electron_selection(events, campaign=self._campaign)
        # combine muon, electron to leptons per event
        good_leptons = ak.with_name(
            ak.concatenate([good_muons, good_electrons], axis=1),
            "PtEtaPhiMCandidate",
        )

        # jet selection
        good_jets = jet_selection(events, campaign=self._campaign)
        good_jets = good_jets[nano_object_overlap(good_jets, good_leptons, dr=0.4)]

        # MET selection
        met_filter_mask = met_selection(events, campaign=self._campaign)

        # === event selection ===

        # combine lepton into dilepton pairs
        lepton_combinations = ak.combinations(
            good_leptons[..., :2],
            n=2,
            replacement=False,
            axis=-1,
            fields=["l0", "l1"],
        )
        dilep = ak.firsts(lepton_combinations.l0 + lepton_combinations.l1)
        dilep_mass = fill_none(dilep.mass, np.nan)

        # z mass, pt cuts
        zll_mass_min = dilep_mass >= 25.0  # originally 12.0
        zll_pt_cut = fill_none(dilep.pt) >= 25.0  # originally 10.0
        # around z mass
        zll_mass_cut = np.abs(dilep_mass - 91.1876) <= 10.0
        # outside of Z mass
        zll_mass_veto = np.abs(dilep_mass - 91.1876) > 10.0

        # leptons should have opposite charge
        ll_opp_charge = fill_none(dilep.charge) == 0

        # min dr between leptons
        dr_ll = min_dr(good_leptons[..., :2])
        dr_ll_cut = dr_ll > 0.2

        # leading lepton pt cuts
        leading_muon_pt_cut = fill_none(ak.firsts(good_muons.pt)) > 25.0
        leading_electron_pt_cut = fill_none(ak.firsts(good_electrons.pt)) > 25.0

        # met cut
        met = events.PuppiMET
        met_pt_cut = met.pt > 30.0

        # two jets
        two_jets = ak.num(good_jets) == 2

        # NOTE: no z diamond mask

        # create masks
        lepton_region_cut_HF = reduce_and(met_pt_cut, zll_mass_veto)
        lepton_region_cut_LF = reduce_and(~met_pt_cut, zll_mass_cut, zll_pt_cut)

        # trigger
        if self.channel == "mumu":
            channel_mask = reduce_and(
                HLT_helper(events, triggers[self.channel]),  # trigger
                leading_muon_pt_cut,  # leading muon pt
                ak.num(good_muons) == 2,  # number of muons
                ak.num(good_electrons) == 0,  # number of electrons
                zll_mass_min,  # dilepton mass min
                ll_opp_charge,  # opposite charge
                dr_ll_cut,  # delta R between leptons
            )
        elif self.channel == "ee":
            channel_mask = reduce_and(
                HLT_helper(events, triggers[self.channel]),
                leading_electron_pt_cut,
                ak.num(good_electrons) == 2,
                ak.num(good_muons) == 0,
                zll_mass_min,
                ll_opp_charge,
                dr_ll_cut,
            )
        elif self.channel == "emu":
            channel_mask = reduce_and(
                HLT_helper(events, triggers[self.channel]),
                (leading_electron_pt_cut | leading_muon_pt_cut),
                ak.num(good_electrons) == 1,
                ak.num(good_muons) == 1,
                zll_mass_min,
                ll_opp_charge,
                dr_ll_cut,
            )
        else:
            raise NotImplementedError(f"{self.channel=} not implemented yet")

        # combine all masks
        common_mask = reduce_and(met_filter_mask, lumi_mask, two_jets)
        event_level_mask = reduce_and(common_mask, channel_mask)

        # === apply masks ===
        pruned_events = events[event_level_mask]

        if len(pruned_events) == 0:
            if self.isArray:
                raise NotImplementedError
            return {dataset: output}

        pruned_events["SelJet"] = good_jets[event_level_mask][:, :2]
        pruned_events["njet"] = ak.count(good_jets[event_level_mask].pt, axis=1)
        pruned_events["dR_jets"] = pruned_events.SelJet[:, 0].delta_r(
            pruned_events.SelJet[:, 1]
        )

        if self.channel == "mumu":
            pruned_events["SelMuon"] = good_muons[event_level_mask][:, :2]
        elif self.channel == "ee":
            pruned_events["SelElectron"] = good_electrons[event_level_mask][:, :2]
        elif self.channel == "emu":
            pruned_events["SelMuon"] = good_muons[event_level_mask][:, :1]
            pruned_events["SelElectron"] = good_electrons[event_level_mask][:, :1]

        pruned_events["dilepton"] = dilep[event_level_mask]
        pruned_events["lepton_region_cut_HF"] = lepton_region_cut_HF[event_level_mask]
        pruned_events["lepton_region_cut_LF"] = lepton_region_cut_LF[event_level_mask]

        # # =====================
        # # tag and probe
        # # =====================

        for i_tag_jet, i_probe_jet in [(0, 1), (1, 0)]:
            tag_jet = pruned_events.SelJet[:, i_tag_jet]
            for btagger in self.histogram_config["btaggers"]:
                # warnings.warn("B-tag score filling not implemented yet.")
                b_score_tag = fill_none(tag_jet[btagger])

                # assign to regions based on b-tag of tag jet
                tagger_name = TAGGER_NAMES[btagger]
                btag_dict = btag_wp_dict[f"{self._year}_{self._campaign}"][tagger_name][
                    "b"
                ]
                tag_jet_cut = {
                    "HF": b_score_tag >= btag_dict["M"],
                    "LF": b_score_tag < btag_dict["L"],
                }

                # create masks for regions
                hf_mask = tag_jet_cut["HF"] & lepton_region_cut_HF[event_level_mask]
                lf_mask = tag_jet_cut["LF"] & lepton_region_cut_LF[event_level_mask]
                assert not ak.any(hf_mask & lf_mask), "HF and LF regions overlap!"

                pruned_events[f"region_{btagger}_{i_probe_jet}"] = ak.where(
                    hf_mask, "HF", ak.where(lf_mask, "LF", "None")
                )

        # =====================
        # weights
        # =====================
        weights = weight_manager(pruned_events, self.SF_map, self.isSyst)
        if is_nominal:
            # we only need JEC shifts, no weight systematics
            systematics = ["nominal"]
        else:
            systematics = [shift_name]

        if not self.noHist:
            histograms = define_and_fill_histograms(
                histogram_config=self.histogram_config,
                events=pruned_events,
                systematics=systematics,
                is_syst=self.isSyst,
                weights=weights,
            )
            output["histograms"] = histograms

        # =====================
        # write arrays
        # =====================
        if self.isArray:
            raise NotImplementedError("Array output not implemented yet")

        return {dataset: output}


if __name__ == "__main__":
    import uproot
    from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

    filename = "root://cmsdcache-kit-disk.gridka.de:1094//store/mc/Run3Summer23BPixNanoAODv12/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/130X_mcRun3_2023_realistic_postBPix_v2-v3/2560000/edcc614d-8bbe-4cd5-91b8-d5c2e82bb1fc.root"
    with uproot.open(filename) as file:
        chunk_size = 10000
        events = NanoEventsFactory.from_root(
            file,
            entry_stop=chunk_size,
            schemaclass=NanoAODSchema,
            metadata={"dataset": "TTto2L2Nu", "filename": filename},
        ).events()

        p = BTagIterativeSFProcessor(
            year="2023",
            campaign="Summer23",
            output_directory="./output",
            isSyst="False",
            isArray=False,
            noHist=False,
            chunksize=chunk_size,
            channel="mumu",
        )
        output = p.process(events)

        histograms = output["TTto2L2Nu"]["histograms"]
        btag = histograms["btag_scores"]

        from IPython import embed

        embed()
