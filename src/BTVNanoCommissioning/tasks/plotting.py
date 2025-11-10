import logging
import warnings
from pathlib import Path

import law
import luigi
import luigi.util
from coffea.util import load

from BTVNanoCommissioning.helpers.xs_scaler import collate, scaleSumW
from BTVNanoCommissioning.tasks.base import BaseTask
from BTVNanoCommissioning.tasks.plotting_utils import (
    VARIABLES,
    HistogramHelper,
    plot_data_mc_histograms,
    plot_MC_histograms_separately,
)
from BTVNanoCommissioning.tasks.processing import ProcessDatasets, processingparameters


def _is_data_sample(name: str) -> bool:
    lowered = name.lower()
    return "run" in lowered or "data" in lowered or "double" in lowered


def _load_histogram_outputs(targets: dict) -> dict[str, dict]:
    logger = logging.getLogger(__name__)
    outputs: dict[str, dict] = {"mc": {}, "data": {}}

    for target in targets.values():
        file_path = Path(target.path)
        loaded = load(file_path)

        for sample_name, payload in loaded.items():
            if sample_name == "xsection":
                continue

            group_key = "data" if _is_data_sample(sample_name) else "mc"
            if sample_name in outputs[group_key]:
                logger.warning(
                    "Duplicate sample %s detected while loading histograms (overwriting previous content)",
                    sample_name,
                )
            outputs[group_key][sample_name] = payload

    return outputs


class plottingparameters(luigi.Config):
    """
    Configuration class for plotting parameters.
    """

    lumi = luigi.FloatParameter(
        default=0.0,
        description="Integrated luminosity in pb^-1 used to scale MC histograms",
    )
    plot_format = luigi.Parameter(
        default="png", description="Output image format (png, pdf, ...)"
    )
    com = luigi.FloatParameter(
        default=13.6, description="Center-of-mass energy in TeV for plot labelling"
    )
    dataset_jsons_base = luigi.Parameter(
        default="",
        description=(
            "Optional base directory prepended to entries in dataset_jsons when resolving JSON paths."
        ),
    )
    dataset_jsons = law.CSVParameter(
        description=(
            "JSON files describing the datasets to process before plotting. "
            "Provide paths relative to the repository root or absolute paths."
        ),
    )


@luigi.util.inherits(processingparameters)
@luigi.util.inherits(plottingparameters)
class PlotDataMC(BaseTask):
    def requires(self):
        return ProcessDatasets.req(self)

    def run(self): ...


@luigi.util.inherits(processingparameters)
@luigi.util.inherits(plottingparameters)
class PlotIterativeResults(BaseTask):
    def _dataset_entries(self) -> list[tuple[str, Path, str]]:
        entries: list[tuple[str, Path, str]] = []
        seen_labels: dict[str, int] = {}

        for raw_entry in self.dataset_jsons:
            entry_path = Path(str(raw_entry))
            if not entry_path.is_absolute() and self.dataset_jsons_base:
                entry_path = Path(self.dataset_jsons_base) / entry_path

            if entry_path.suffix != ".json":
                if entry_path.suffix:
                    entry_path = entry_path.with_suffix(".json")
                else:
                    entry_path = entry_path.parent / f"{entry_path.name}.json"

            json_path = entry_path
            base_label = json_path.stem
            seen_count = seen_labels.get(base_label, 0)
            seen_labels[base_label] = seen_count + 1

            label = base_label if seen_count == 0 else f"{base_label}_{seen_count}"
            output_name = f"hists_{label}.coffea"

            entries.append((label, json_path, output_name))

        return entries

    def requires(self):
        requirements: dict = {}
        for label, json_path, output_name in self._dataset_entries():
            requirements[label] = ProcessDatasets.req(
                self,
                json=str(json_path),
                output_file=output_name,
            )
        return requirements

    def store_parts(self) -> tuple[str, ...]:
        base_parts = super().store_parts()
        dataset_part = "-".join(label for label, _, _ in self._dataset_entries())
        return (*base_parts, self.campaign, self.workflow, dataset_part)

    def output(self):
        return self.local_directory_target("plot")

    def run(self):
        logger = logging.getLogger(self.__class__.__name__)
        # log_level = getattr(logging, str(self.log_level).upper(), logging.INFO)
        log_level = logging.INFO
        logging.basicConfig(
            level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        # logger.setLevel(log_level)

        dataset_inputs = self.input()

        dataset_entries = self._dataset_entries()
        ordered_inputs: dict[str, luigi.target.Target] = {}
        for label, json_path, _ in dataset_entries:
            try:
                ordered_inputs[label] = dataset_inputs[label]
            except KeyError as exc:
                raise RuntimeError(
                    f"Missing required input target for dataset {label}"
                ) from exc
            else:
                logger.debug(
                    "Loading histograms for %s from %s (JSON: %s)",
                    label,
                    ordered_inputs[label].path,
                    json_path,
                )

        outputs = _load_histogram_outputs(ordered_inputs)
        mc_outputs = outputs.get("mc", {})
        data_outputs = outputs.get("data", {})
        if not data_outputs or not mc_outputs:
            logger.warning("Missing MC or data histograms; skipping plot creation")
            return

        if self.lumi:
            mc_outputs = scaleSumW(mc_outputs, self.lumi)

        merged_outputs = {**mc_outputs, **data_outputs}
        mergemap = {
            "data": list(data_outputs.keys()),
            "mc": list(mc_outputs.keys()),
        }
        collated = collate(merged_outputs, mergemap)

        plot_root = Path(self.output().path)
        plot_root.mkdir(parents=True, exist_ok=True)

        lumi_label = f"{self.lumi / 1000:.1f} fb^-1" if self.lumi else ""

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message=".*encountered in divide.*"
            )
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message=".*meaningful error bars.*"
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*List indexing selection is experimental.*",
            )

            for variable in VARIABLES:
                for channel in ("ee", "emu", "mumu", "incl"):
                    key = f"{variable}_{channel}"
                    try:
                        mc_hist = collated["mc"][key]
                        data_hist = collated["data"][key]
                    except KeyError:
                        logger.debug("Skipping missing histogram key %s", key)
                        continue

                    mc_helper = HistogramHelper(mc_hist, is_data=False)
                    data_helper = HistogramHelper(data_hist, is_data=True)

                    save_path = plot_root / variable / channel
                    save_path.mkdir(parents=True, exist_ok=True)

                    regions = mc_helper.get_axis("region")

                    for region in regions:
                        common = {
                            "mc_histogram": mc_helper,
                            "data_histogram": data_helper,
                            "region": region,
                            "channel": channel,
                            "save_path": save_path,
                            "com": self.com,
                            "lumi_label": lumi_label,
                            "plot_format": self.plot_format,
                        }

                        plot_data_mc_histograms(**common, split_flavor=False)
                        plot_data_mc_histograms(**common, split_flavor=True)
                        plot_MC_histograms_separately(
                            mc_helper,
                            region,
                            channel,
                            save_path,
                            self.com,
                            lumi_label,
                            eta_bin=sum,
                            plot_format=self.plot_format,
                        )

                        eta_axis = mc_helper.get_axis("eta")
                        for eta_bin in range(len(eta_axis)):
                            plot_data_mc_histograms(
                                **common, split_flavor=False, eta_bin=eta_bin
                            )
                            plot_data_mc_histograms(
                                **common, split_flavor=True, eta_bin=eta_bin
                            )
                            plot_MC_histograms_separately(
                                mc_helper,
                                region,
                                channel,
                                save_path,
                                self.com,
                                lumi_label,
                                eta_bin=eta_bin,
                                plot_format=self.plot_format,
                            )

                        jet_axis = mc_helper.get_axis("jet_index")
                        for jet_index in range(len(jet_axis)):
                            plot_data_mc_histograms(
                                **common,
                                split_flavor=False,
                                jet_index=jet_index,
                            )
                            plot_data_mc_histograms(
                                **common, split_flavor=True, jet_index=jet_index
                            )

        self.output().touch()
