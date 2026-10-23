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
        description="Executor to fetch datasets",
    )
    fetch_workers = luigi.IntParameter(
        default=4,
        description="Number of workers to fetch datasets (for futures executor)",
    )
    verbose = luigi.BoolParameter(
        default=False,
        description="verbose flag for fetch.py script",
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
            self.fetch_executor,
            "--workers",
            str(self.fetch_workers)
        ]
        if self.skipvalidation:
            cmd.append("--skipvalidation")
        if self.overwrite:
            cmd.append("--overwrite")
        if self.verbose:
            cmd.append("--verbose")

        subprocess.run(cmd, check=True)

    def output(self):
        return law.LocalFileTarget(
            os.path.abspath(os.path.join("metadata", self.campaign, self.json))
        )
