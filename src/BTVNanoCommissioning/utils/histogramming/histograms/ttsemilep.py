import hist as Hist


def get_histograms(axes, **kwargs):
    hists = {}

    c_ttsemilep = kwargs.get("c_ttsemilep", None)
    if c_ttsemilep == None:
        raise ValueError(
            "c_ttsemilep is not specified when running ttsemilep workflow."
        )

    hists["dr_cjet"] = Hist.Hist(
        axes["syst"], axes["flav"], axes["dr"], Hist.storage.Weight()
    )
    if not c_ttsemilep:
        for i in range(4):
            hists[f"dr_mujet{i}"] = Hist.Hist(
                axes["syst"], axes["flav"], axes["dr"], Hist.storage.Weight()
            )
    for i in ["mu"]:
        hists[f"{i}_pfRelIso04_all"] = Hist.Hist(
            axes["syst"], axes["iso"], Hist.storage.Weight()
        )
        hists[f"{i}_dxy"] = Hist.Hist(axes["syst"], axes["dxy"], Hist.storage.Weight())
        hists[f"{i}_dz"] = Hist.Hist(axes["syst"], axes["dz"], Hist.storage.Weight())

    for jet_index in range(2):
        hists[f"btagUParTAK4B_{jet_index}"] = Hist.Hist(
            axes["syst"],
            axes["flav"],
            Hist.axis.Regular(
                100,
                0.0,
                1,
                name="discr",
                label=f"btagUParTAK4B, jet {jet_index}",
            ),
            Hist.storage.Weight(),
        )
        hists[f"btagUParTAK4B_noSF_{jet_index}"] = Hist.Hist(
            axes["syst"],
            axes["flav"],
            Hist.axis.Regular(
                100,
                0.0,
                1,
                name="discr",
                label=f"btagUParTAK4B (no SF), jet {jet_index}",
            ),
            Hist.storage.Weight(),
        )

        for pt_bin in [(0, 50), (50, 80), (80, "inf")]:
            hists[f"btagUParTAK4B_pt{pt_bin[0]}to{pt_bin[1]}_{jet_index}"] = Hist.Hist(
                axes["syst"],
                axes["flav"],
                Hist.axis.Regular(
                    100,
                    0.0,
                    1,
                    name="discr",
                    label=f"btagUParTAK4B, jet {jet_index} pt {pt_bin[0]} to {pt_bin[1]}",
                ),
                Hist.storage.Weight(),
            )
            hists[f"btagUParTAK4B_pt{pt_bin[0]}to{pt_bin[1]}_noSF_{jet_index}"] = Hist.Hist(
                axes["syst"],
                axes["flav"],
                Hist.axis.Regular(
                    100,
                    0.0,
                    1,
                    name="discr",
                    label=f"btagUParTAK4B (no SF), jet {jet_index} pt {pt_bin[0]} to {pt_bin[1]}",
                ),
                Hist.storage.Weight(),
            )

    return hists
