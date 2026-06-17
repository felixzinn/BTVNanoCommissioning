import os

import law
import luigi
import luigi.util

from BTVNanoCommissioning.tasks.processing import ProcessBTagIterative
from BTVNanoCommissioning.workflows.btag_ttbar_iterative.plot_script import main

law.contrib.load("wlcg")


@luigi.util.requires(ProcessBTagIterative)
class PlotBtagIterative(law.Task):
    def output(self):
        return {
            "local": law.LocalDirectoryTarget(
                os.path.join(
                    os.environ.get("BTV_OUTPUT_DIR", "."),
                    self.version,
                    self.__class__.__name__,
                )
            ),
            "eos": law.wlcg.WLCGDirectoryTarget(
                os.path.join(
                    "f/fzinn/BTV/btag_sf",
                    self.version,
                    self.__class__.__name__,
                ),
                fs="wlcg_fs",
            ),
        }

    @law.decorator.safe_output
    def run(self):
        inp = self.input()
        out = self.output()
        input_files = ",".join(target.abspath for target in inp["coffea"])
        lumi = sum(float(target["lumi"].load()) for target in inp["lumi"])
        main(
            input_paths=input_files,
            output_path=out["local"].path,
            lumi=lumi,
        )

        out["eos"].copy_from_local(out["local"].path)
        
