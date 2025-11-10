"""Reusable plotting helpers extracted from the iterative b-tagging script."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import hist
import matplotlib.pyplot as plt
import mplhep
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from BTVNanoCommissioning.utils.plot_utils import MCerrorband, plotratio

# Shared plotting metadata
VARIABLES = {"btagDeepFlavB", "btagUParTAK4B", "btagRobustParTAK4B", "njet"}
FLAVOR_LABELS = {"c": [4], "b": [5], "l": [0, 1, 6]}
COLORS = {"b": "#5790fc", "c": "#f89c20", "l": "#964a8b"}
COLOR_LIST = [COLORS[key] for key in FLAVOR_LABELS]
CHANNEL_LABELS = {
    "mumu": r"$\mu\mu$",
    "ee": r"$ee$",
    "emu": r"$e\mu$",
    "incl": "Inclusive",
}


def set_yaxis_limit_ratio(ax: Axes, max_upper: float = 2.0) -> None:
    """Clamp the ratio y-axis between zero and a configurable upper bound."""

    lower, upper = ax.get_ylim()
    ax.set_ylim(
        max(0.0, lower),
        min(max_upper, math.ceil(10.0 * upper) / 10.0),
    )


def define_figure(
    com: float, lumi_label: float, ratio: bool = True
) -> tuple[Figure, tuple[Axes, Axes | None]]:
    """Create a CMS-styled figure optionally including a ratio subplot."""

    if ratio:
        fig, (ax_main, ax_ratio) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(10, 10),
        )
        fig.subplots_adjust(hspace=0.03, top=0.92, bottom=0.1, right=0.97)
    else:
        fig, ax_main = plt.subplots(figsize=(10, 8))
        ax_ratio = None

    mplhep.cms.label(
        "Preliminary", com=com, lumi=lumi_label, ax=ax_main, loc=0, data=True
    )
    return fig, (ax_main, ax_ratio)


def configure_plot(
    fig: Figure,
    ax: Axes,
    region: str,
    channel: str,
    ax_ratio: Axes | None = None,
    xlabel: str | None = None,
) -> None:
    """Apply shared axis ranges, labels, title, and legend formatting."""

    if "btag" in ax.get_xlabel().lower():
        ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=max(1e-1, ax.get_ylim()[0]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.legend(loc="upper center", ncols=3)

    if ax_ratio is not None:
        ax.set_xlabel(None)
        ax_ratio.set_ylabel("Data/MC")
        set_yaxis_limit_ratio(ax_ratio)

    fig.suptitle(f"{region} region, {CHANNEL_LABELS.get(channel, channel)} channel")
    fig.tight_layout(pad=0.75)


def add_space_on_top(ax: Axes, log: bool = False, fraction: float = 0.2) -> None:
    """Add headroom to the y-axis to avoid clipping legend entries."""

    lower, upper = ax.get_ylim()
    if log:
        lower_log = math.log10(lower)
        upper_log = math.log10(upper)
        new_upper = 10 ** (lower_log + (upper_log - lower_log) / (1 - fraction))
    else:
        new_upper = lower + (upper - lower) / (1 - fraction)
    ax.set_ylim(top=new_upper)


def save_figure(
    fig: Figure, ax: Axes, save_path: Path, filename: str, plot_format: str = "png"
) -> None:
    """Persist the figure in both linear and logarithmic y-scales."""

    save_path = Path(save_path)
    (save_path / "lin").mkdir(parents=True, exist_ok=True)
    (save_path / "log").mkdir(parents=True, exist_ok=True)

    add_space_on_top(ax, log=False)
    fig.savefig(save_path / "lin" / f"{filename}.{plot_format}", dpi=300)

    try:
        ax.set_yscale("log")
        add_space_on_top(ax, log=True)
    except ValueError:
        # Negative bin contents can prevent log scaling; skip silently.
        pass
    else:
        fig.savefig(save_path / "log" / f"{filename}.{plot_format}", dpi=300)
    finally:
        plt.close(fig)


class HistogramHelper:
    """Convenience wrapper around a hist.Hist for repeated slicing/plotting."""

    def __init__(self, histogram: hist.Hist, *, is_data: bool = False) -> None:
        self.histogram = histogram
        self.is_data = is_data
        self.axes = histogram.axes
        self.axis_names = [axis.name for axis in histogram.axes]
        self.value_axis_name = self.axis_names[-1] if self.axis_names else None
        self._flav_axis = (
            histogram.axes["flav"]
            if (not is_data and "flav" in self.axis_names)
            else None
        )
        self.label = histogram.label

    def iter_axis_labels(self, axis_name: str):
        """Iterate over categorical axis labels, yielding display names."""

        axis = self.get_axis(axis_name)
        if axis is None:
            raise ValueError(f"Axis {axis_name} does not exist on this histogram")

        identifiers = getattr(axis, "identifiers", None)
        if identifiers is None:
            raise ValueError(
                f"Axis {axis_name} does not expose categorical identifiers"
            )

        for identifier in identifiers:
            yield getattr(identifier, "name", identifier)

    def get_axis(self, axis_name: str):
        """Return the axis by name, or None if it does not exist."""

        if axis_name in self.axis_names:
            return self.axes[axis_name]
        return None

    def has_axis(self, axis_name: str) -> bool:
        """Check whether the histogram exposes the requested axis."""

        return axis_name in self.axis_names

    def get_summed(
        self, region: str, channel: str, eta_bin: Any, jet_index: Any
    ) -> hist.Hist:
        """Return the histogram summed over flavours for a given selection."""

        selection = self._build_selection(
            region=region,
            channel=channel,
            eta_bin=eta_bin,
            jet_index=jet_index,
        )
        return self.histogram[selection]

    def get_by_flavor(
        self,
        region: str,
        channel: str,
        *,
        jet_index: Any = sum,
        eta_bin: Any = sum,
    ) -> list[hist.Hist]:
        """Return a list of per-flavour histograms for the given selection."""

        if self.is_data:
            raise ValueError("Data histograms do not carry flavour information")
        if not self.has_axis("flav"):
            raise ValueError("Histogram does not contain a flavour axis")

        flavour_indices: Iterable[Sequence[int]] = FLAVOR_LABELS.values()
        results: list[hist.Hist] = []
        for indices in flavour_indices:
            overrides = {
                "flav": list(self._flav_axis.index(indices)),
                "region": region,
                "channel": channel,
                "eta": eta_bin,
                "jet_index": jet_index,
            }
            selection = self._build_selection(**overrides)
            hist_view = self.histogram[selection]
            if "flav" in hist_view.axes.name:
                hist_view = hist_view[{"flav": sum}]
            results.append(hist_view)
        return results

    def plot_error_band(
        self,
        ax: Axes,
        region: str,
        channel: str,
        *,
        jet_index: Any = sum,
        eta_bin: Any = sum,
    ) -> None:
        """Draw the nominal MC uncertainty band."""

        if self.is_data:
            raise ValueError("Cannot plot uncertainties for data histograms")
        MCerrorband(
            self.get_summed(
                region=region, channel=channel, eta_bin=eta_bin, jet_index=jet_index
            ),
            ax=ax,
        )

    def plot_histogram(
        self,
        ax: Axes,
        region: str,
        channel: str,
        *,
        jet_index: Any = sum,
        eta_bin: Any = sum,
        split_flavor: bool = False,
        stack: bool = True,
    ) -> None:
        """Render the histogram as either a summed or flavour-separated plot."""

        if self.is_data:
            mplhep.histplot(
                self.get_summed(
                    region=region, channel=channel, eta_bin=eta_bin, jet_index=jet_index
                ),
                label="Data",
                histtype="errorbar",
                color="black",
                yerr=True,
                ax=ax,
            )
            return

        if split_flavor and self.has_axis("flav"):
            mplhep.histplot(
                self.get_by_flavor(
                    region=region, channel=channel, jet_index=jet_index, eta_bin=eta_bin
                ),
                label=list(FLAVOR_LABELS.keys()),
                stack=stack,
                histtype="fill" if stack else "step",
                yerr=True,
                color=COLOR_LIST,
                ax=ax,
            )
            return

        mplhep.histplot(
            self.get_summed(
                region=region, channel=channel, eta_bin=eta_bin, jet_index=jet_index
            ),
            label="MC",
            histtype="fill" if stack else "step",
            yerr=True,
            color="#f89c20",
            ax=ax,
        )

    def _build_selection(
        self,
        *,
        region: Any = None,
        channel: Any = None,
        eta_bin: Any = None,
        jet_index: Any = None,
        syst: Any = None,
        flav: Any = None,
        **additional_overrides: Any,
    ) -> dict[str, Any]:
        """Create a mapping suitable for hist indexing based on axis availability."""

        selection: dict[str, Any] = {}

        for axis_name in self.axis_names:
            if axis_name == "syst":
                selection[axis_name] = syst if syst is not None else "nominal"
            elif axis_name == "flav":
                selection[axis_name] = flav if flav is not None else sum
            elif axis_name == "region":
                selection[axis_name] = region if region is not None else sum
            elif axis_name == "channel":
                selection[axis_name] = channel if channel is not None else sum
            elif axis_name in {"eta", "abs_eta"}:
                selection[axis_name] = eta_bin if eta_bin is not None else sum
            elif axis_name in {"jet_index", "jet"}:
                selection[axis_name] = jet_index if jet_index is not None else sum
            elif axis_name == self.value_axis_name:
                selection[axis_name] = additional_overrides.get(axis_name, slice(None))
            else:
                selection[axis_name] = additional_overrides.get(axis_name, sum)

        return selection


def plot_data_mc_histograms(
    mc_histogram: HistogramHelper,
    data_histogram: HistogramHelper,
    region: str,
    channel: str,
    save_path: Path,
    com: float,
    lumi_label: str,
    *,
    split_flavor: bool = False,
    eta_bin: Any = sum,
    jet_index: Any = sum,
    plot_format: str = "png",
) -> None:
    """Plot a data-vs-MC comparison with an optional flavour split."""

    with rc_context(mplhep.style.CMS):
        if split_flavor and not mc_histogram.has_axis("flav"):
            return

        fig, (ax, ax_ratio) = define_figure(com=com, lumi_label=lumi_label)

        mc_histogram.plot_histogram(
            ax=ax,
            region=region,
            channel=channel,
            jet_index=jet_index,
            eta_bin=eta_bin,
            split_flavor=split_flavor,
        )
        data_histogram.plot_histogram(
            ax=ax,
            region=region,
            channel=channel,
            jet_index=jet_index,
            eta_bin=eta_bin,
        )

        mc_histogram.plot_error_band(
            ax=ax,
            region=region,
            channel=channel,
            jet_index=jet_index,
            eta_bin=eta_bin,
        )

        ax_ratio = plotratio(
            data_histogram.get_summed(
                region=region, channel=channel, jet_index=jet_index, eta_bin=eta_bin
            ),
            mc_histogram.get_summed(
                region=region, channel=channel, jet_index=jet_index, eta_bin=eta_bin
            ),
            ax=ax_ratio,
        )

        configure_plot(
            fig=fig, ax=ax, region=region, channel=channel, ax_ratio=ax_ratio
        )

        filename_parts: list[str] = []
        if jet_index is not None and jet_index is not sum:
            filename_parts.append(f"jet{jet_index}")
        if eta_bin is not None and eta_bin is not sum:
            filename_parts.append(f"etabin{eta_bin}")
        label = "stacked" if split_flavor else "summed"
        filename_parts.append(f"{region}_histograms_{label}")
        filename = "_".join(filename_parts)

    save_figure(
        fig=fig, ax=ax, save_path=save_path, filename=filename, plot_format=plot_format
    )


def plot_MC_histograms_separately(
    mc_histogram: HistogramHelper,
    region: str,
    channel: str,
    save_path: Path,
    com: float,
    lumi_label: str,
    *,
    eta_bin: Any = sum,
    plot_format: str = "png",
) -> None:
    """Plot the MC components as separate lines without data or ratio panels."""

    with rc_context(mplhep.style.CMS):
        fig, (ax, _) = define_figure(com=com, lumi_label=lumi_label, ratio=False)
        mc_histogram.plot_histogram(
            ax=ax,
            region=region,
            channel=channel,
            jet_index=sum,
            eta_bin=eta_bin,
            split_flavor=mc_histogram.has_axis("flav"),
            stack=False,
        )
        configure_plot(
            fig=fig, ax=ax, region=region, channel=channel, xlabel=mc_histogram.label
        )

        filename = f"{region}_histograms_separate"
        if eta_bin is not None and eta_bin is not sum:
            filename = f"etabin{eta_bin}_{filename}"

    save_figure(
        fig=fig, ax=ax, save_path=save_path, filename=filename, plot_format=plot_format
    )
