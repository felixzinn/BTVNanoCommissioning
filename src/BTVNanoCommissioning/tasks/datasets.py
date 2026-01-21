import os
import subprocess

import law
import luigi
import luigi.util

from BTVNanoCommissioning.tasks.base import baseparameters


@luigi.util.inherits(baseparameters)
class FetchDatasets(law.Task):
    json = luigi.Parameter(
        description="json output filename",
    )
    definition = luigi.Parameter(
        description="Dataset definition file",
    )
    blacklist_sites = luigi.Parameter(
        default=[],
        description="List of sites to blacklist",
    )
    skipvalidation = luigi.BoolParameter(
        default=False,
        description="Skip file validation",
    )
    overwrite = luigi.BoolParameter(
        default=False,
        description="Overwrite existing json file",
    )
    fetch_executor = luigi.ChoiceParameter(
        default="futures",
        choices=("futures", "iterative"),
        description="Executor to fetch datasets"
    )

    def run(self):
        cmd = [
            "python",
            "scripts/fetch.py",
            "--campaign",
            self.campaign,
            "--year",
            self.year,
            "--input",
            self.definition,
            "--blacklist_sites",
            self.blacklist_sites,
            "--output",
            self.json,
            "--executor",
            self.fetch_executor
        ]
        if self.skipvalidation:
            cmd.append("--skipvalidation")
        if self.overwrite:
            cmd.append("--overwrite")

        subprocess.run(cmd, check=True)

    def output(self):
        return law.LocalFileTarget(
            os.path.abspath(os.path.join("metadata", self.campaign, self.json))
        )
