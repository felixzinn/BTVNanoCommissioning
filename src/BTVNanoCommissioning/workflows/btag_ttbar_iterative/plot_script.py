import argparse
import pathlib
import warnings

from matplotlib.axis import Axis
import mplhep


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
    ax.autoscale()
    mplhep.ylow(ax=ax)
    mplhep.yscale_legend(ax=ax)


def main(input_paths, output_path, lumi):
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
    output = {key: value for path in input_paths for key, value in load(path).items()}
    output.pop("xsection")

    # Set up output directory
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Clear existing plots in output directory
    clear_plots(output_path)

    # scale with sumw and lumi
    output = scaleSumW(output, lumi=lumi)
    # merge samples
    merge_map = sample_mergemap | {
        "mc": [sample for sample in output.keys() if "Run" not in sample],
        "data": [sample for sample in output.keys() if "Run" in sample],
    }
    collated = collate(output, merge_map)

    # samples in output
    samples = set(collated.keys()) - {"data", "mc"}

    # Loop over variables and plot
    non_plottable = {}
    for variable, histogram in collated["mc"].items():
        if not isinstance(histogram, Hist):
            continue

        variable_dir = output_path / variable
        variable_dir.mkdir(parents=True, exist_ok=True)

        mc_histograms: dict[str, Hist] = {
            sample: collated[sample][variable]
            for sample in samples
            if collated[sample] is not None
        }
        data_hist = collated["data"][variable]

        # plot 1D histograms, i.e. kinematics
        if mc_histograms and next(iter(mc_histograms.values())).ndim == 1:
            fig, ax = plot_1D_histogram(
                mc_histograms=mc_histograms, data=data_hist, lumi=lumi
            )
            fig.savefig(variable_dir / "mc_data.png")
            ylog_scale(ax)
            fig.savefig(variable_dir / "mc_data_log.png")
            plt.close(fig)

        # btag has multiple axis
        elif "btag" in variable:
            # sum over eta, pt, jet_index; loop over regions
            # index is (systematic, flavor, eta, pt, region, jet_index, btag_discriminant)
            for region in data_hist.axes["region"]:
                data_hist_region = data_hist["nominal", sum, sum, sum, region, sum, :]

                # split by sample
                mc_sum = {
                    sample: hist["nominal", sum, sum, sum, region, sum, :]
                    for sample, hist in mc_histograms.items()
                }
                fig_suptitle = f"{region} region"
                fig, ax = plot_1D_histogram(
                    mc_histograms=mc_sum,
                    data=data_hist_region,
                    suptitle=fig_suptitle,
                    lumi=lumi,
                )
                fig.savefig(variable_dir / f"data_mc_{region}.png")
                ylog_scale(ax)
                fig.savefig(variable_dir / f"data_mc_{region}_log.png")
                plt.close(fig)

                # split by flavor
                # c=4, b=5, light=0,1,6
                mc_flavor = {
                    "b": sum(
                        hist["nominal", hist_lib.loc(5), sum, sum, region, sum, :]
                        for hist in mc_histograms.values()
                    ),
                    "c": sum(
                        hist["nominal", hist_lib.loc(4), sum, sum, region, sum, :]
                        for hist in mc_histograms.values()
                    ),
                    "l": sum(
                        hist[
                            "nominal",
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

                fig, ax = plot_1D_histogram(
                    mc_histograms=mc_flavor,
                    data=data_hist_region,
                    suptitle=fig_suptitle,
                    lumi=lumi,
                )
                fig.savefig(variable_dir / f"data_mc_{region}_byflav.png")
                ylog_scale(ax)
                fig.savefig(variable_dir / f"data_mc_{region}_byflav_log.png")
                plt.close(fig)
        else:
            non_plottable += {variable}

    if non_plottable:
        warnings.warn(f"Some histograms were not plottable: {non_plottable}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(input_paths=args.input, output_path=args.output, lumi=args.lumi)
