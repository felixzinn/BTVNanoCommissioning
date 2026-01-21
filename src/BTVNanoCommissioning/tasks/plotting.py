import os

import law
import luigi
import luigi.util

from BTVNanoCommissioning.tasks.processing import ProcessBTagIterative
from BTVNanoCommissioning.workflows.btag_ttbar_iterative.plot_script import main


@luigi.util.inherits(ProcessBTagIterative)
class PlotBtagIterative(law.Task):
    lumi = luigi.FloatParameter(
        description="Luminosity in pb^-1 for scaling",
    )

    def requires(self):
        return ProcessBTagIterative.req(self)

    def output(self):
        return law.LocalDirectoryTarget(
            os.path.join(
                os.environ.get("BTV_OUTPUT_DIR", "."),
                self.version,
                self.__class__.__name__,
            )
        )

    def run(self):
        input_files = ",".join(inp.abspath for inp in self.input())
        main(
            input_paths=input_files,
            output_path=self.output().path,
            lumi=self.lumi,
        )
