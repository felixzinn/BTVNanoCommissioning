import argparse
import itertools
import pathlib
import warnings

import mplhep
from matplotlib.axis import Axis

plot_style = mplhep.style.CMS | {
    "figure.constrained_layout.use": True,
}


def clear_plots(directory):
    directory = pathlib.Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            clear_plots(item)
            item.rmdir()
        else:
            item.unlink()


parser = argparse.ArgumentParser(
    description="Plot histograms for btag_ttbar_iterative workflow"
)
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the input coffea file(s) (comma-separated for multiple files)",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Directory to save the output plots",
)
parser.add_argument("--lumi", type=float, help="Luminosity in pb^-1")


def ylog_scale(ax: Axis):
    ax.set_yscale("log")
    # ax.autoscale()
    mplhep.set_ylow(ax=ax)
    mplhep.yscale_legend(ax=ax, soft_fail=True)


def main(input_paths, output_path, lumi, only_btag: bool = False):
    import pathlib

    import hist as hist_lib
    import matplotlib.pyplot as plt
    from coffea.util import load
    from hist import Hist

    from BTVNanoCommissioning.helpers.xs_scaler import collate, scaleSumW
    from BTVNanoCommissioning.utils.plot_utils import sample_mergemap
    from BTVNanoCommissioning.workflows.btag_ttbar_iterative.plotting import (
        plot_1D_histogram,
    )

    # Load input files
    input_paths = [pathlib.Path(path.strip()) for path in input_paths.split(",")]
    output = {}
    for path in input_paths:
        _out = load(path)
        # coffea checkpointing writes tuple into output
        if isinstance(_out, dict):
            _out = {key: value for key, value in _out.items() if key != "xsection"}
        elif isinstance(_out, tuple):
            assert "xsection" in _out[0], (
                "xsection not found in coffea output (tuple[0])"
            )
            _out = {key: value for key, value in _out[0].items() if key != "xsection"}
        else:
            raise TypeError(f"Unexpected type for coffea output: {type(_out)}")

        # remove systematics from histograms, keep only nominal
        for sample, histograms in _out.items():
            if sample in output:
                warnings.warn(
                    f"Sample {sample} already exists in output, overwriting",
                    stacklevel=2,
                )
            output[sample] = {"sumw": histograms["sumw"]}  # keep sumw for scaling
            for variable, histogram in histograms.items():
                if isinstance(histogram, Hist):
                    if "syst" in histogram.axes.name:
                        # remove systematics axis if present
                        output[sample][variable] = histogram["nominal", ...]
                    else:
                        output[sample][variable] = histogram

    # Set up output directory
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Clear existing plots in output directory
    clear_plots(output_path)

    # scale with sumw and lumi
    output = scaleSumW(output, lumi=lumi)
    # merge samples
    sample_mergemap.pop("Z+jets", None)  # Z+jets and DY are double
    merge_map = sample_mergemap | {
        "mc": [sample for sample in output if "Run" not in sample],
        "data": [sample for sample in output if "Run" in sample],
    }
    collated = {
        key: value
        for key, value in collate(output, merge_map).items()
        if value is not None
    }

    # samples in output
    samples = set(collated.keys()) - {"data", "mc"}

    # Loop over variables and plot
    non_plottable = {}
    for variable, data_hist in collated["data"].items():
        if only_btag and "btag" not in variable:
            continue
        if not isinstance(data_hist, Hist):
            continue

        mc_histograms: dict[str, Hist] = {
            sample: collated[sample][variable]
            for sample in samples
            if variable in collated[sample]
        }

        if not mc_histograms:
            warnings.warn(
                f"No MC histograms found for variable {variable}", stacklevel=2
            )
            continue

        variable_dir = output_path / variable
        variable_dir.mkdir(parents=True, exist_ok=True)

        ndim_hist = next(iter(mc_histograms.values())).ndim
        # plot 1D histograms, i.e. kinematics
        if mc_histograms and ndim_hist == 1:
            fig, (ax, rax) = plot_1D_histogram(
                mc_histograms=mc_histograms, data=data_hist, lumi=lumi
            )
            fig.savefig(variable_dir / "mc_data.png")
            ylog_scale(ax)
            fig.savefig(variable_dir / "mc_data_log.png")
            plt.close(fig)

        elif ndim_hist == 2:
            # 2d histograms have region as additional axis
            # loop over regions
            for region in data_hist.axes["region"]:
                data_hist_region = data_hist[:, region]

                # split by sample
                mc_sum = {
                    sample: hist[:, region] for sample, hist in mc_histograms.items()
                }
                fig_suptitle = f"{region} region"
                fig, (ax, rax) = plot_1D_histogram(
                    mc_histograms=mc_sum,
                    data=data_hist_region,
                    suptitle=fig_suptitle,
                    lumi=lumi,
                )
                fig.savefig(variable_dir / f"data_mc_{region}.png")
                ylog_scale(ax)
                fig.savefig(variable_dir / f"data_mc_{region}_log.png")
                plt.close(fig)

        # btag has multiple axis
        elif "btag" in variable:
            # sum over eta, pt, jet_index; loop over regions
            # index is (flavor, eta, pt, region, jet_index, btag_discriminant)
            for region in data_hist.axes["region"]:
                if "dR_jets" in data_hist.axes.name:
                    # index is (flavor, eta, pt, dR, region, jet_index, btag_discriminant)
                    data_hist_region = data_hist[sum, sum, sum, sum, region, sum, :]
                    # split by sample
                    mc_sum = {
                        sample: hist[sum, sum, sum, sum, region, sum, :]
                        for sample, hist in mc_histograms.items()
                    }
                    # split by flavor
                    # c=4, b=5, light=0,1,6
                    mc_flavor = {
                        "b": sum(
                            hist[hist_lib.loc(5), sum, sum, sum, region, sum, :]
                            for hist in mc_histograms.values()
                        ),
                        "c": sum(
                            hist[hist_lib.loc(4), sum, sum, sum, region, sum, :]
                            for hist in mc_histograms.values()
                        ),
                        "l": sum(
                            hist[
                                [hist_lib.loc(v) for v in (0, 1, 6)],
                                sum,
                                sum,
                                sum,
                                region,
                                sum,
                                :,
                            ][sum, :]
                            for hist in mc_histograms.values()
                        ),
                    }
                else:
                    data_hist_region = data_hist[sum, sum, sum, region, sum, :]
                    # split by sample
                    mc_sum = {
                        sample: hist[sum, sum, sum, region, sum, :]
                        for sample, hist in mc_histograms.items()
                    }
                    # split by flavor
                    # c=4, b=5, light=0,1,6
                    mc_flavor = {
                        "b": sum(
                            hist[hist_lib.loc(5), sum, sum, region, sum, :]
                            for hist in mc_histograms.values()
                        ),
                        "c": sum(
                            hist[hist_lib.loc(4), sum, sum, region, sum, :]
                            for hist in mc_histograms.values()
                        ),
                        "l": sum(
                            hist[
                                [hist_lib.loc(v) for v in (0, 1, 6)],
                                sum,
                                sum,
                                region,
                                sum,
                                :,
                            ][sum, :]
                            for hist in mc_histograms.values()
                        ),
                    }

                fig_suptitle = f"{region} region"
                fig, (ax, rax) = plot_1D_histogram(
                    mc_histograms=mc_sum,
                    data=data_hist_region,
                    suptitle=fig_suptitle,
                    lumi=lumi,
                )
                fig.savefig(variable_dir / f"data_mc_{region}.png")
                ylog_scale(ax)
                fig.savefig(variable_dir / f"data_mc_{region}_log.png")
                plt.close(fig)

                fig, (ax, _) = plot_1D_histogram(
                    mc_histograms=mc_flavor,
                    data=data_hist_region,
                    suptitle=fig_suptitle,
                    lumi=lumi,
                )
                fig.savefig(variable_dir / f"data_mc_{region}_byflav.png")
                ylog_scale(ax)
                fig.savefig(variable_dir / f"data_mc_{region}_byflav_log.png")
                plt.close(fig)

                fig, (ax, _) = plot_1D_histogram(
                    mc_histograms=mc_flavor,
                    data=data_hist_region,
                    suptitle=fig_suptitle,
                    lumi=lumi,
                    stack=False,
                )
                fig.savefig(variable_dir / f"data_mc_{region}_byflav_nostack.png")
                ylog_scale(ax)
                fig.savefig(variable_dir / f"data_mc_{region}_byflav_nostack_log.png")
                plt.close(fig)

                # add flavour ratio plots
                fig, (ax, *raxs) = plt.subplots(
                    4,
                    1,
                    gridspec_kw={"height_ratios": [3, 1, 1, 1]},
                    sharex=True,
                    figsize=(8, 10),
                )
                plot_1D_histogram(
                    mc_histograms=mc_flavor,
                    data=data_hist_region,
                    suptitle=fig_suptitle,
                    lumi=lumi,
                    ax=ax,
                    plot_ratio=False,
                )
                ylog_scale(ax)
                xlabel = ax.get_xlabel()
                for i, (flav, mc_hist) in enumerate(mc_flavor.items()):
                    rax = raxs[i]
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action="ignore",
                            message=".*invalid value encountered in true_divide.*",
                        )
                        bin_centers = mc_hist.axes[0].centers
                        ratio = mc_hist.values() / sum(mc_flavor.values()).values()
                        rax.plot(
                            bin_centers,
                            ratio,
                            linestyle="none",
                            marker=".",
                            color="black",
                        )
                        rax.grid(which="major")
                        rax.set_ylabel(f"{flav} / sum")
                        rax.set_ylim(bottom=0)

                raxs[-1].set_xlabel(xlabel)
                ax.set_xlabel(None)
                fig.savefig(variable_dir / f"data_mc_{region}_byflav_ratios.png")
                for rax in raxs:
                    rax.set_yscale("log")
                    rax.autoscale()
                fig.savefig(variable_dir / f"data_mc_{region}_byflav_ratios_log.png")
                plt.close(fig)

                # # plot per pT/eta bin
                # for pt, eta in itertools.product(
                #     data_hist.axes["pt"], data_hist.axes["eta"]
                # ):
                #     if "dR_jets" in data_hist.axes.name:
                #         data_hist_region = data_hist[
                #             sum,
                #             hist_lib.loc(eta[0]),
                #             hist_lib.loc(pt[0]),
                #             sum,
                #             region,
                #             sum,
                #             :,
                #         ]
                #         mc_sum = {
                #             sample: hist[
                #                 sum,
                #                 hist_lib.loc(eta[0]),
                #                 hist_lib.loc(pt[0]),
                #                 sum,
                #                 region,
                #                 sum,
                #                 :,
                #             ]
                #             for sample, hist in mc_histograms.items()
                #         }
                #         mc_flavor = {
                #             "b": sum(
                #                 hist[
                #                     hist_lib.loc(5),
                #                     hist_lib.loc(eta[0]),
                #                     hist_lib.loc(pt[0]),
                #                     sum,
                #                     region,
                #                     sum,
                #                     :,
                #                 ]
                #                 for hist in mc_histograms.values()
                #             ),
                #             "c": sum(
                #                 hist[
                #                     hist_lib.loc(4),
                #                     hist_lib.loc(eta[0]),
                #                     hist_lib.loc(pt[0]),
                #                     sum,
                #                     region,
                #                     sum,
                #                     :,
                #                 ]
                #                 for hist in mc_histograms.values()
                #             ),
                #             "l": sum(
                #                 hist[
                #                     [hist_lib.loc(v) for v in (0, 1, 6)],
                #                     hist_lib.loc(eta[0]),
                #                     hist_lib.loc(pt[0]),
                #                     sum,
                #                     region,
                #                     sum,
                #                     :,
                #                 ][sum, :]
                #                 for hist in mc_histograms.values()
                #             ),
                #         }
                #     else:
                #         data_hist_region = data_hist[
                #             sum, hist_lib.loc(eta[0]), hist_lib.loc(pt[0]), region, sum, :
                #         ]
                #         mc_sum = {
                #             sample: hist[
                #                 sum, hist_lib.loc(eta[0]), hist_lib.loc(pt[0]), region, sum, :
                #             ]
                #             for sample, hist in mc_histograms.items()
                #         }
                #         mc_flavor = {
                #             "b": sum(
                #                 hist[
                #                     hist_lib.loc(5),
                #                     hist_lib.loc(eta[0]),
                #                     hist_lib.loc(pt[0]),
                #                     region,
                #                     sum,
                #                     :,
                #                 ]
                #                 for hist in mc_histograms.values()
                #             ),
                #             "c": sum(
                #                 hist[
                #                     hist_lib.loc(4),
                #                     hist_lib.loc(eta[0]),
                #                     hist_lib.loc(pt[0]),
                #                     region,
                #                     sum,
                #                     :,
                #                 ]
                #                 for hist in mc_histograms.values()
                #             ),
                #             "l": sum(
                #                 hist[
                #                     [hist_lib.loc(v) for v in (0, 1, 6)],
                #                     hist_lib.loc(eta[0]),
                #                     hist_lib.loc(pt[0]),
                #                     region,
                #                     sum,
                #                     :,
                #                 ][sum, :]
                #                 for hist in mc_histograms.values()
                #             ),
                #         }

                #     eta = "to".join(str(e) for e in eta)
                #     pt = "to".join(str(p) for p in pt)
                #     fig_suptitle = f"{region} region"
                #     fig, (ax, _) = plot_1D_histogram(
                #         mc_histograms=mc_sum,
                #         data=data_hist_region,
                #         suptitle=fig_suptitle,
                #         lumi=lumi,
                #     )
                #     fig.savefig(variable_dir / f"data_mc_{region}_eta{eta}_pT{pt}.png")
                #     ylog_scale(ax)
                #     fig.savefig(variable_dir / f"data_mc_{region}_eta{eta}_pT{pt}_log.png")
                #     plt.close(fig)

                #     fig, (ax, _) = plot_1D_histogram(
                #         mc_histograms=mc_flavor,
                #         data=data_hist_region,
                #         suptitle=fig_suptitle,
                #         lumi=lumi,
                #     )
                #     fig.savefig(
                #         variable_dir / f"data_mc_{region}_eta{eta}_pT{pt}_byflav.png"
                #     )
                #     ylog_scale(ax)
                #     fig.savefig(
                #         variable_dir
                #         / f"data_mc_{region}_eta{eta}_pT{pt}_byflav_log.png"
                #     )
                #     plt.close(fig)
                # plot per pT/eta bin
                for eta in data_hist.axes["eta"]:
                    if "dR_jets" in data_hist.axes.name:
                        data_hist_region = data_hist[
                            sum,
                            hist_lib.loc(eta[0]),
                            sum,
                            sum,
                            region,
                            sum,
                            :,
                        ]
                        mc_sum = {
                            sample: hist[
                                sum,
                                hist_lib.loc(eta[0]),
                                sum,
                                sum,
                                region,
                                sum,
                                :,
                            ]
                            for sample, hist in mc_histograms.items()
                        }
                        mc_flavor = {
                            "b": sum(
                                hist[
                                    hist_lib.loc(5),
                                    hist_lib.loc(eta[0]),
                                    sum,
                                    sum,
                                    region,
                                    sum,
                                    :,
                                ]
                                for hist in mc_histograms.values()
                            ),
                            "c": sum(
                                hist[
                                    hist_lib.loc(4),
                                    hist_lib.loc(eta[0]),
                                    sum,
                                    sum,
                                    region,
                                    sum,
                                    :,
                                ]
                                for hist in mc_histograms.values()
                            ),
                            "l": sum(
                                hist[
                                    [hist_lib.loc(v) for v in (0, 1, 6)],
                                    hist_lib.loc(eta[0]),
                                    sum,
                                    sum,
                                    region,
                                    sum,
                                    :,
                                ][sum, :]
                                for hist in mc_histograms.values()
                            ),
                        }
                    else:
                        data_hist_region = data_hist[
                            sum, hist_lib.loc(eta[0]), sum, region, sum, :
                        ]
                        mc_sum = {
                            sample: hist[sum, hist_lib.loc(eta[0]), sum, region, sum, :]
                            for sample, hist in mc_histograms.items()
                        }
                        mc_flavor = {
                            "b": sum(
                                hist[
                                    hist_lib.loc(5),
                                    hist_lib.loc(eta[0]),
                                    sum,
                                    region,
                                    sum,
                                    :,
                                ]
                                for hist in mc_histograms.values()
                            ),
                            "c": sum(
                                hist[
                                    hist_lib.loc(4),
                                    hist_lib.loc(eta[0]),
                                    sum,
                                    region,
                                    sum,
                                    :,
                                ]
                                for hist in mc_histograms.values()
                            ),
                            "l": sum(
                                hist[
                                    [hist_lib.loc(v) for v in (0, 1, 6)],
                                    hist_lib.loc(eta[0]),
                                    sum,
                                    region,
                                    sum,
                                    :,
                                ][sum, :]
                                for hist in mc_histograms.values()
                            ),
                        }

                    eta = "to".join(str(e) for e in eta)
                    fig_suptitle = f"{region} region"
                    fig, (ax, _) = plot_1D_histogram(
                        mc_histograms=mc_sum,
                        data=data_hist_region,
                        suptitle=fig_suptitle,
                        lumi=lumi,
                    )
                    fig.savefig(variable_dir / f"data_mc_{region}_eta{eta}.png")
                    ylog_scale(ax)
                    fig.savefig(variable_dir / f"data_mc_{region}_eta{eta}_log.png")
                    plt.close(fig)

                    fig, (ax, _) = plot_1D_histogram(
                        mc_histograms=mc_flavor,
                        data=data_hist_region,
                        suptitle=fig_suptitle,
                        lumi=lumi,
                    )
                    fig.savefig(variable_dir / f"data_mc_{region}_eta{eta}_byflav.png")
                    ylog_scale(ax)
                    fig.savefig(
                        variable_dir / f"data_mc_{region}_eta{eta}_byflav_log.png"
                    )
                    plt.close(fig)
        else:
            non_plottable += {variable}

    if non_plottable:
        warnings.warn(f"Some histograms were not plottable: {non_plottable}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(input_paths=args.input, output_path=args.output, lumi=args.lumi)
