import warnings

import matplotlib.pyplot as plt
import mplhep
from hist import Hist
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from BTVNanoCommissioning.utils.plot_utils import plotratio

plot_style = mplhep.style.CMS | {
    "figure.constrained_layout.use": True,
}


def set_symmetric_limits(ax: Axis, min_y: float = 0.0, max_y: float = 2.0):
    """Set symmetric y-limits around 1 for ratio plots."""
    if min_y >= max_y:
        raise ValueError("min_y must be less than max_y")
    if min_y >= 1.0 or max_y <= 1.0:
        raise ValueError("Limits must enclose 1.0 to be symmetric around it.")

    # Get current limits and clip to provided bounds
    curr_lim = ax.get_ylim()
    new_min = max(min_y, curr_lim[0])
    new_max = min(max_y, curr_lim[1])

    # Calculate symmetric limits around 1.0
    diff = max(1.0 - new_min, new_max - 1.0)
    ax.set_ylim(1.0 - diff, 1.0 + diff)


def plot_1D_histogram(
    mc_histograms: dict[str, Hist],
    data: Hist = None,
    lumi: float = None,
    suptitle: str = None,
) -> tuple[Figure, Axis]:
    with plt.rc_context(plot_style):
        fig: Figure
        ax: Axis
        rax: Axis

        if data is not None:
            fig, (ax, rax) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
            )
        else:
            fig, ax = plt.subplots()
        mplhep.cms.label(
            label="Preliminary",
            data=data is not None,
            loc=0,
            ax=ax,
            lumi=lumi / 1000,
            lumi_format="{:.4f}",
            com="13.6",
        )
        mplhep.histplot(
            list(mc_histograms.values()),
            ax=ax,
            stack=True,
            label=mc_histograms.keys(),
            histtype="fill",
            yerr=True,
            sort="yield",
            # color=[]
            # alpha=[]
        )
        if data is not None:
            mplhep.histplot(
                data, ax=ax, histtype="errorbar", color="black", label="data"
            )
        # MCErrorband()

        # styling
        ax.legend(ncols=2, loc="upper center")
        ax.set_ylabel("Events")
        if data is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    category=RuntimeWarning,
                    message=r"invalid value encountered in divide",
                )
                warnings.filterwarnings(
                    action="ignore",
                    category=RuntimeWarning,
                    message=r"All sumw are zero!",
                )
                rax = plotratio(data, sum(mc_histograms.values()), ax=rax)
            rax.set_xlabel(ax.get_xlabel())
            ax.set_xlabel(None)
            rax.set_ylabel("Data/MC")

            # set y-limits for ratio plot
            set_symmetric_limits(rax, min_y=0.0, max_y=2.0)

        if suptitle is not None:
            fig.suptitle(suptitle)

        # style adjustments
        # NOTE: should switch to mplhep >= v1.0.0.rc4
        mplhep.ylow(ax=ax)
        mplhep.yscale_legend(ax=ax)

    return fig, ax
