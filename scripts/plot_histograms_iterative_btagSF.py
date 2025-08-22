"""Plot Histograms for iterative b-tagging SFs.
need to be in the directory of the histograms, i.e. contains
- hists_MC/hists_MC.coffea
- hists_data/hists_data.coffea
"""

import argparse
import logging
import os
from pathlib import Path
import math
import warnings

from coffea.util import load

from BTVNanoCommissioning.helpers.xs_scaler import scaleSumW, collate
from BTVNanoCommissioning.utils.plot_utils import MCerrorband, plotratio

import mplhep
import hist
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import rc_context


# ================
# setup logging
# ================
logger = logging.getLogger(__name__)


# ================
# setup arguments
# ================
parser = argparse.ArgumentParser(
    description="Plot histograms for iterative b-tagging SFs"
)
parser.add_argument(
    "--lumi",
    help="luminosity in pb^-1",
    type=float,
)
parser.add_argument(
    "--log-level",
    help="Set the logging level",
    default="INFO",
    type=str.upper,  # Convert to uppercase automatically
)


# ==============
# some defaults
# ==============
VARIABLES = {"btagDeepFlavB", "btagPNetB", "btagUParTAK4B", "btagRobustParTAK4B"}
FLAVOR_LABELS = {"c": [4], "b": [5], "l": [0, 1, 6]}
COLORS = {
    "b": "#5790fc",
    "c": "#f89c20",
    "l": "#964a8b",
}
COLOR_LIST = [COLORS[flav] for flav in FLAVOR_LABELS.keys()]
CHANNEL_LABELS = {
    "mumu": r"$\mu\mu$",
    "ee": r"$ee$",
    "emu": r"$e\mu$",
    "incl": "Inclusive",
}


# ========================
# define helper functions
# ========================
def set_yaxis_limit_ratio(ax: Axes, max_upper: float = 2) -> None:
    """Set the y-axis limit for the ratio plot.

    Sets the y-axis limit between 0 and max_upper, taking the current y-axis
    limit and setting:
    - the lower limit to 0 if it is below 0
    - the upper limit rounded up to the nearest 0.1
    - if it is above max_upper, set to max_upper

    Args:
        ax: The axes to set the limit for
        max_upper: The maximum upper limit for the y-axis, defaults to 2
    """
    ax.set_ylim(
        max(0, ax.get_ylim()[0]),
        min(max_upper, math.ceil(10 * ax.get_ylim()[1]) / 10),
    )


def define_figure(
    com: float, lumi_label: str, ratio: bool = True
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Define the figure and axes for plotting.

    Creates matplotlib figure with appropriate layout for physics plots,
    including CMS label and luminosity information.

    Args:
        com: Center-of-mass energy in TeV
        lumi_label: Luminosity label in fb^-1
        ratio: Whether to include a ratio subplot

    Returns:
        Tuple containing the figure and a tuple of (main_axes, ratio_axes).
        If ratio=False, ratio_axes will be None.
    """
    if ratio:
        fig, (ax, ax_ratio) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(10, 10),
        )
        fig.subplots_adjust(hspace=0.03, top=0.92, bottom=0.1, right=0.97)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax_ratio = None
    mplhep.cms.label("Preliminary", com=com, lumi=lumi_label, ax=ax, loc=0, data=True)
    return fig, (ax, ax_ratio)


def configure_plot(
    fig: Figure,
    ax: Axes,
    region: str,
    channel: str,
    ax_ratio: Axes = None,
    xlabel: str = None,
) -> None:
    """Configure plot appearance and layout.

    Sets axis limits, labels, legends, and overall plot formatting.

    Args:
        fig: The matplotlib figure
        ax: Main plot axes
        region: Physics region (e.g., "HF", "LF")
        channel: Lepton channel (e.g., "ee", "mumu", "emu")
        ax_ratio: Ratio plot axes, optional
        xlabel: X-axis label, optional
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=max(1e-1, ax.get_ylim()[0]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.legend(loc="upper center", ncols=3)

    if ax_ratio is not None:
        ax.set_xlabel(None)
        ax_ratio.set_ylabel("Data/MC")
        set_yaxis_limit_ratio(ax_ratio)

    fig.suptitle(f"{region} region, {CHANNEL_LABELS[channel]} channel")
    fig.tight_layout(pad=0.75)


def add_space_on_top(ax: Axes, log: bool = False, fraction: float = 0.2) -> None:
    down, up = ax.get_ylim()
    if log:
        up_log = math.log10(up)
        down_log = math.log10(down)
        delta = up_log - down_log
        new_upper = 10 ** (down_log + 1 / (1 - fraction) * delta)  # (up_log + fraction * delta)
    else:
        new_upper = down + 1 / (1 - fraction) * (up - down)  # up + fraction * (up - down)
    ax.set_ylim(top=new_upper)


def save_figure(
    fig: Figure,
    ax: Axes,
    save_path: os.PathLike,
    filename: str,
    plot_format: str = "png",
) -> None:
    """Save the figure in both linear and log scale.

    Saves the plot twice: once with linear y-scale and once with log y-scale,
    in separate subdirectories.

    Args:
        fig: The matplotlib figure to save
        ax: Main plot axes (used for setting log scale)
        save_path: Directory path where to save the plots
        filename: Base filename (without extension)
    """
    add_space_on_top(ax, log=False)
    fig.savefig(save_path / "lin" / f"{filename}.{plot_format}", dpi=300)
    try:
        ax.set_yscale("log")
        add_space_on_top(ax, log=True)
    except ValueError as e:
        # log scale might fail if negative entries (due to weights)
        logging.warning(f"Failed to save log scale plot: {e}")
    else:
        fig.savefig(save_path / "log" / f"{filename}.{plot_format}", dpi=300)
    finally:
        plt.close(fig)


class HistogramHelper:
    """Helper class to simplify histogram access patterns.

    Provides convenient methods for accessing and plotting histogram data,
    handling differences between data and MC histograms automatically.

    Attributes:
        histogram: The underlying hist.Hist object
        is_data: Whether this represents data (True) or MC (False)
        axes: Histogram axes for easy access
        label: Histogram label
    """

    def __init__(self, histogram: hist.Hist, is_data: bool = False):
        """Initialize the histogram helper.

        Args:
            histogram: The hist.Hist object to wrap
            is_data: Whether this is data (True) or MC (False)
        """
        self.histogram = histogram
        self.is_data = is_data
        self._flav_axis = histogram.axes["flav"] if not is_data else None
        self.axes = histogram.axes
        self.label = histogram.label

    def get_summed(self, region: str, channel: str, jet_index: sum) -> hist.Hist:
        """Get histogram summed over all flavors.

        Args:
            region: Physics region to select
            channel: Lepton channel to select

        Returns:
            Histogram summed over all jet flavors
        """
        # if self.is_data:
        #     return self.histogram["nominal", 0, sum, sum, region, sum, :]
        if "channel" in self.histogram.axes.name:
            return self.histogram[
                "nominal", sum, sum, sum, region, channel, jet_index, :
            ]
        return self.histogram["nominal", sum, sum, sum, region, jet_index, :]

    def get_by_flavor(
        self, region: str, channel: str, jet_index=sum
    ) -> list[hist.Hist]:
        """Get list of histograms separated by flavor.

        Args:
            region: Physics region to select
            channel: Lepton channel to select

        Returns:
            List of histograms, one for each jet flavor

        Raises:
            ValueError: If called on data histogram (flavors not available)
        """
        if self.is_data:
            raise ValueError("Cannot separate data by flavor")

        if "channel" in self.histogram.axes.name:
            return [
                self.histogram[
                    "nominal",
                    list(self._flav_axis.index(flavor)),
                    sum,
                    sum,
                    region,
                    channel,
                    jet_index,
                    :,
                ][sum, :]
                for flavor in FLAVOR_LABELS.values()
            ]
        return [
            self.histogram[
                "nominal",
                list(self._flav_axis.index(flavor)),
                sum,
                sum,
                region,
                # channel,
                jet_index,
                :,
            ][sum, :]
            for flavor in FLAVOR_LABELS.values()
        ]

    def plot_error_band(
        self, ax: Axes, region: str, channel: str, jet_index=sum
    ) -> None:
        """Plot MC error band on the given axes.

        Args:
            ax: Matplotlib axes to plot on
            region: Physics region to select
            channel: Lepton channel to select

        Raises:
            ValueError: If called on data histogram
        """
        if self.is_data:
            raise ValueError("Cannot plot error band for data")
        MCerrorband(
            self.get_summed(region=region, channel=channel, jet_index=jet_index), ax=ax
        )

    def plot_histogram(
        self,
        ax: Axes,
        region: str,
        channel: str,
        jet_index=sum,
        split_flavor: bool = False,
        stack: bool = True,
    ) -> None:
        """Plot the histogram on the given axes.

        Args:
            ax: Matplotlib axes to plot on
            region: Physics region to select
            channel: Lepton channel to select
            split_flavor: Whether to separate by jet flavor (MC only)
            stack: Whether to stack flavors (only used if split_flavor=True)
        """
        if self.is_data:
            mplhep.histplot(
                self.get_summed(region=region, channel=channel, jet_index=jet_index),
                label="Data",
                histtype="errorbar",
                color="black",
                yerr=True,
                ax=ax,
            )
        else:
            if split_flavor:
                mplhep.histplot(
                    self.get_by_flavor(
                        region=region, channel=channel, jet_index=jet_index
                    ),
                    label=list(FLAVOR_LABELS.keys()),
                    stack=stack,
                    histtype="fill" if stack else "step",
                    yerr=True,
                    ax=ax,
                    color=COLOR_LIST,
                )
            else:
                mplhep.histplot(
                    self.get_summed(
                        region=region, channel=channel, jet_index=jet_index
                    ),
                    label="MC",
                    histtype="fill",
                    yerr=True,
                    ax=ax,
                    color="#f89c20",
                )


def plot_data_mc_histograms(
    mc_histogram: HistogramHelper,
    data_histogram: HistogramHelper,
    region: str,
    channel: str,
    save_path: os.PathLike,
    com: float,
    lumi_label: str,
    split_flavor: bool = False,
    jet_index=sum,
):
    """Plot data vs MC comparison with ratio plot.

    Creates a two-panel plot with data/MC comparison on top and
    data/MC ratio on bottom. Can show MC either summed or split by flavor.

    Args:
        mc_histogram: MC histogram helper
        data_histogram: Data histogram helper
        region: Physics region to plot
        channel: Lepton channel to plot
        save_path: Directory to save plots
        com: Center-of-mass energy in TeV
        lumi_label: Luminosity in fb^-1
        split_flavor: Whether to show MC split by flavor (stacked)
    """
    with rc_context(mplhep.style.CMS):
        fig, (ax, ax_ratio) = define_figure(com=com, lumi_label=lumi_label)

        if split_flavor:
            mc_histogram.plot_histogram(
                ax=ax,
                region=region,
                channel=channel,
                split_flavor=True,
                jet_index=jet_index,
            )
        else:
            mc_histogram.plot_histogram(
                ax=ax,
                region=region,
                channel=channel,
                split_flavor=False,
                jet_index=jet_index,
            )

        data_histogram.plot_histogram(
            ax=ax, region=region, channel=channel, jet_index=jet_index
        )
        mc_histogram.plot_error_band(
            region=region, channel=channel, ax=ax, jet_index=jet_index
        )

        # plot ratio
        ax_ratio = plotratio(
            data_histogram.get_summed(
                region=region, channel=channel, jet_index=jet_index
            ),
            mc_histogram.get_summed(
                region=region, channel=channel, jet_index=jet_index
            ),
            ax=ax_ratio,
        )

        configure_plot(
            fig=fig, ax=ax, region=region, channel=channel, ax_ratio=ax_ratio
        )
        if split_flavor:
            filename = f"{region}_histograms_stacked"
        else:
            filename = f"{region}_histograms_summed"
        if jet_index != sum:
            filename = f"jet{jet_index}_{filename}"

        save_figure(
            fig=fig,
            ax=ax,
            save_path=save_path,
            filename=filename,
        )


def plot_MC_histograms_separately(
    mc_histogram: HistogramHelper,
    region: str,
    channel: str,
    save_path: os.PathLike,
    com: float,
    lumi_label: str,
):
    """Plot MC histograms by flavor as separate lines.

    Creates a single-panel plot showing each jet flavor as a separate
    line (not stacked). No data or ratio plot included.

    Args:
        mc_histogram: MC histogram helper
        region: Physics region to plot
        channel: Lepton channel to plot
        save_path: Directory to save plots
        com: Center-of-mass energy in TeV
        lumi_label: Luminosity in fb^-1
    """
    with rc_context(mplhep.style.CMS):
        fig, (ax, _) = define_figure(com=com, lumi_label=lumi_label, ratio=False)
        mc_histogram.plot_histogram(
            ax=ax, region=region, channel=channel, split_flavor=True, stack=False
        )
        configure_plot(
            fig=fig, ax=ax, region=region, channel=channel, xlabel=mc_histogram.label
        )
        save_figure(
            fig=fig,
            ax=ax,
            save_path=save_path,
            filename=f"{region}_histograms_separate",
        )


def load_histograms():
    """Load histogram files for Monte Carlo (MC) and data samples.

    Returns:
        dict: A dictionary mapping 'mc' and 'data' to their respective coffea output.

    Raises:
        FileNotFoundError: If any of the required histogram files are missing.
    """
    files = {
        "mc": [
            Path(f"hists_MC_{process}/hists_MC_{process}.coffea")
            for process in ("dy", "ttbar", "WZ", "singletop")
        ],
        "data": [Path("hists_data/hists_data.coffea")],
    }

    logger.debug("Checking for histogram files")
    missing_files = [
        path for paths in files.values() for path in paths if not path.exists()
    ]
    if missing_files:
        logger.error(f"Missing histogram files: {missing_files}")
        raise FileNotFoundError(f"Missing histogram files: {missing_files}")

    logger.info("Loading histogram files")
    result = {}
    for name, paths in files.items():
        output = {}
        for path in paths:
            logger.debug(f"Loading {name} histograms from {path}")
            output |= load(path)
            logger.debug(f"Successfully loaded {name} histograms")
        result[name] = output
    logger.info("All histogram files loaded successfully")
    return result


def setup_logging(log_level: str):
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(numeric_level)


def main():
    """Main function to create all plots.

    Parses command line arguments, loads histograms, scales by luminosity,
    and creates all comparison plots for each variable, channel, and region.
    """
    # parse arguments and set defaults
    args = parser.parse_args()

    # Configure logging based on command line argument
    setup_logging(args.log_level)

    lumi = args.lumi  # pb^-1
    lumi_label = (
        lumi / 1000
    )  # fb^-1, hardcoded in mplhep, https://github.com/scikit-hep/mplhep/blob/master/src/mplhep/label.py#L356
    COM = 13.6

    logger.info(
        f"Starting plot generation with luminosity: {lumi} pb^-1 ({lumi_label:.1f} fb^-1)"
    )

    output = load_histograms()
    # pop data from output to avoid memory usage in scaling
    data_output = output.pop("data", None)
    data_output.pop("xsection", None)  # remove xsection if exists
    output["mc"].pop("xsection", None)  # remove xsection if exists

    logger.info("Preparing histograms for plotting")
    # scale the histograms by the luminosity
    logger.debug("Scaling histograms by luminosity")
    output = scaleSumW(output, lumi)
    logger.debug("Adding data back to output")
    for key in list(data_output.keys()):
        output[key] = data_output.pop(key)  # add data histograms back to output

    # merge datasets
    logger.debug("Collating datasets")
    mergemap = {
        "data": [m for m in output.keys() if "Run" in m],
        "mc": [m for m in output.keys() if "Run" not in m],
    }
    logger.debug(f"Collating with mergemap: {mergemap}")
    collated = collate(output, mergemap)

    logger.info("Start plotting")

    # ================
    # plot histograms
    # ================

    # see the btag_iterative_sf workflow for the structure of the histograms

    with warnings.catch_warnings():
        # data/MC ratio ignore if empty MC bins
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*encountered in divide.*",
        )
        # plotting errorbars of MC, if empty bins -> ignore
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message=".*meaningful error bars.*"
        )
        # indexing hists with list
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*List indexing selection is experimental.*",
        )

        for variable in VARIABLES:
            logger.info(f"Plotting histograms for variable: {variable}")
            for channel in ("ee", "emu", "mumu", "incl"):
                key = f"{variable}_{channel}"
                try:
                    mc_histogram = collated["mc"][key]
                    data_histogram = collated["data"][key]
                except KeyError:
                    logger.warning(
                        f"Key {key} not found in the histograms.",
                    )
                    continue

                logger.debug(f"Plotting histograms for {variable=}, {channel=}")

                mc_histogram = HistogramHelper(mc_histogram, is_data=False)
                data_histogram = HistogramHelper(data_histogram, is_data=True)

                # for channel in mc_histogram.axes["channel"]:
                save_path = Path("plot", variable, channel)
                # Create all subdirectories at once
                (save_path / "lin").mkdir(parents=True, exist_ok=True)
                (save_path / "log").mkdir(parents=True, exist_ok=True)

                for region in mc_histogram.axes["region"]:
                    common_args = {
                        "mc_histogram": mc_histogram,
                        "region": region,
                        "channel": channel,
                        "save_path": save_path,
                        "com": COM,
                        "lumi_label": lumi_label,
                    }

                    logger.debug(
                        "Plotting histograms for variable "
                        f"{variable}, region {region}, channel {channel}"
                    )

                    plot_data_mc_histograms(
                        **common_args, data_histogram=data_histogram, split_flavor=False
                    )
                    plot_data_mc_histograms(
                        **common_args, data_histogram=data_histogram, split_flavor=True
                    )

                    # MC separately doesn't need data
                    plot_MC_histograms_separately(**common_args)

                    # plot jets separated
                    for jet_index in (0, 1):
                        common_args["jet_index"] = jet_index
                        plot_data_mc_histograms(
                            **common_args,
                            data_histogram=data_histogram,
                            split_flavor=False,
                        )
                        plot_data_mc_histograms(
                            **common_args,
                            data_histogram=data_histogram,
                            split_flavor=True,
                        )


if __name__ == "__main__":
    main()
