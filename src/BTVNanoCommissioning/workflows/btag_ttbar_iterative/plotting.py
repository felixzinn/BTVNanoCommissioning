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
            lumi=lumi,
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

        if suptitle is not None:
            fig.suptitle(suptitle)

    return fig, ax
