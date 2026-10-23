import os

import law
import luigi
import luigi.util

from BTVNanoCommissioning.tasks.processing import ProcessBTagIterative
from BTVNanoCommissioning.workflows.btag_ttbar_iterative.plot_script import main

law.contrib.load("wlcg")


@luigi.util.requires(ProcessBTagIterative)
class PlotBtagIterative(law.Task):
    input_files = law.CSVParameter(
        default=("all"), description="Coffea files to use for plotting (default: all)"
    )
    transfer = luigi.BoolParameter(
        default=True, description="Transfer output to EOS (default: True)"
    )
    only_btag = luigi.BoolParameter(
        default=False, description="Only plot btag variables (default: False)"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_files = sorted(self.input_files)

    def output(self):
        paths = [
            self.version,
            self.__class__.__name__,
            "_".join([file.replace(".coffea", "") for file in self.input_files]),
        ]
        out = {
            "local": law.LocalDirectoryTarget(
                os.path.join(os.environ.get("BTV_OUTPUT_DIR", "."), *paths)
            ),
        }
        if self.transfer:
            out["eos"] = (
                law.wlcg.WLCGDirectoryTarget(
                    os.path.join("f/fzinn/BTV/btag_sf", *paths),
                    fs="wlcg_fs",
                ),
            )
        return out

    @law.decorator.safe_output
    def run(self):
        inp = self.input()
        out = self.output()
        if "all" in self.input_files:
            input_files = ",".join(target.abspath for target in inp["coffea"])
            lumi = sum(float(target["lumi"].load()) for target in inp["lumi"])
        else:
            input_files = ",".join(
                target.abspath
                for target in inp["coffea"]
                if any(input_file in target.basename for input_file in self.input_files)
            )
            lumi = sum(
                float(target["lumi"].load())
                for target in inp["lumi"]
                if any(
                    input_file.replace(".coffea", "_lumi.txt")
                    in target["lumi"].basename
                    for input_file in self.input_files
                )
            )
        if not input_files:
            raise ValueError(
                f"No input files found for {self.input_files} in {inp['coffea']}"
            )
        main(
            input_paths=input_files,
            output_path=out["local"].path,
            lumi=lumi,
            only_btag=self.only_btag,
        )

        if self.transfer:
            out["eos"].copy_from_local(out["local"].path)


@luigi.util.inherits(PlotBtagIterative)
class PlotBtagIterative2024(law.WrapperTask):
    year = "2024"
    campaign = "Summer24"
    # NOTE: need to define definition_files as a Parameter to avoid MissingParameterException
    # will be overwritten in __init__ with the correct list of files
    definition_files = PlotBtagIterative.definition_files.copy(default="")

    def __init__(self, *args, **kwargs):
        self.runs = ["C", "D", "E", "F", "G", "H", "I"]
        self.data_definition_files = [
            f"metadata/{self.campaign}/btag_iterative_sf/data_Run{run}.txt"
            for run in self.runs
        ]
        self.mc = ["ttbar", "singletop", "dy", "WZ"]
        self.mc_definition_files = [
            f"metadata/{self.campaign}/btag_iterative_sf/{mc_file}.txt"
            for mc_file in self.mc
        ]

        super().__init__(
            *args,
            **kwargs,
            definition_files=[
                *self.data_definition_files,
                *self.mc_definition_files,
            ],
        )

    def requires(self):
        return [
            *(
                PlotBtagIterative.req(
                    self,
                    campaign=self.campaign,
                    year=self.year,
                    input_files=[
                        f"data_Run{run}_hists.coffea",
                        *[f"{mc_file}_hists.coffea" for mc_file in self.mc],
                    ],
                )
                for run in self.runs
            ),
            PlotBtagIterative.req(
                self, campaign=self.campaign, year=self.year, input_files=["all"]
            ),
        ]
