import os
import subprocess

import law
import luigi
import luigi.util

from BTVNanoCommissioning.tasks.base import baseparameters
from BTVNanoCommissioning.tasks.datasets import FetchDatasets


class processingparameters(luigi.Config):
    workflow = luigi.Parameter(
        description="Workflow name",
    )
    executor = luigi.Parameter(
        default="futures",
        description="Executor to use for coffea processing",
    )
    limit = luigi.IntParameter(
        default=None,
        description="Limit number of files to process per dataset",
    )
    max_chunks = luigi.IntParameter(
        default=None,
        description="Maximum number of chunks to process per dataset",
    )
    scaleout = luigi.IntParameter(
        default=4,
        description="Scale out for processing",
    )
    chunksize = luigi.IntParameter(
        default=250_000,
        description="Chunk size for processing",
    )
    is_syst = luigi.ChoiceParameter(
        default="False",
        choices=["False", "all", "weight_only", "JERC_split", "JP_MC"],
        description="Whether to run systematic variations",
    )


@luigi.util.inherits(processingparameters, baseparameters)
class ProcessDatasets(law.Task):
    json = luigi.Parameter(
        description="Path to JSON file with dataset and file locations",
    )
    output_file = luigi.Parameter(
        description="Output histogram filename",
    )
    skipbadfiles = luigi.BoolParameter(
        default=False,
        description="Skip bad files during processing",
    )

    def run(self):
        self.output().parent.touch()
        cmd = [
            "python",
            "runner.py",
            "--campaign",
            self.campaign,
            "--year",
            self.year,
            "--workflow",
            self.workflow,
            "--json",
            self.json,
            "--output",
            self.output_file,
            "--outputdir",
            self.output_base,
            "--executor",
            self.executor,
            "--scaleout",
            str(self.scaleout),
            "--chunk",
            str(self.chunksize),
            "--isSyst",
            self.is_syst,
        ]
        if self.limit is not None:
            cmd += ["--limit", str(self.limit)]
        if self.max_chunks is not None:
            cmd += ["--max", str(self.max_chunks)]

        subprocess.run(
            cmd,
            check=True,
        )

    @property
    def output_base(self):
        return os.path.join(
            os.environ.get("BTV_OUTPUT_DIR", "."),
            self.version,
            self.__class__.__name__,
        )

    def output(self):
        return law.LocalFileTarget(
            os.path.join(
                self.output_base,
                os.path.splitext(os.path.basename(self.output_file))[0],
                self.output_file,
            )
        )

    def requires(self):
        stem = os.path.splitext(self.json)[0]
        file_name = []
        for item in reversed(stem.split("/")):
            if item == self.campaign:
                break
            file_name.append(item)
        file_name = os.path.join(*reversed(file_name))
        return FetchDatasets.req(self, json=f"{file_name}.json", definition=f"{stem}.txt")


@luigi.util.inherits(processingparameters, baseparameters)
class ProcessBTagIterative(law.WrapperTask):
    workflow = luigi.Parameter(default="btag_ttbar_iterative_sf")
    definition_files = law.CSVParameter(
        description="List of dataset definition files",
    )

    def requires(self):
        reqs = []
        for dataset_file in self.definition_files:
            stem = os.path.splitext(dataset_file)[0]
            reqs.append(
                ProcessDatasets.req(
                    self,
                    json=f"{stem}.json",
                    output_file=f"{os.path.basename(stem)}_hists.coffea",
                )
            )
        return reqs
