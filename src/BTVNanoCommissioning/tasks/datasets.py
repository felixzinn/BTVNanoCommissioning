import os
from pathlib import Path

import law
import luigi
import luigi.util

from BTVNanoCommissioning.tasks.base import BaseTask
from BTVNanoCommissioning.workflows import workflows


class datasetparameters(luigi.Config):
    """
    Configuration class for dataset parameters.
    """

    inp = luigi.Parameter(
        default=law.NO_STR,
        description="List of samples in DAS",
    )

    out = luigi.Parameter(
        default="test_my_samples.json",
        description="Output file name",
    )

    xrd = luigi.Parameter(
        default=law.NO_STR,
        description="xrootd prefix string otherwise get from available sites",
    )

    from_path = luigi.BoolParameter(
        default=False,
        description="For samples that are not published on DAS. If this option is set then the format of the --input file must be adjusted. It should be: dataset_name path_to_files.",
    )

    from_dataset = luigi.BoolParameter(
        default=False,
        description="Input dataset only",
    )

    from_workflow = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, *sorted(workflows.keys())],
        description="Use the predefined workflows",
    )

    testfile = luigi.BoolParameter(
        default=False,
        description="Construct file list in the test directory. Specify the test directory path, create the json file for individual dataset",
    )

    whitelist_sites = luigi.Parameter(
        default=law.NO_STR,
        description="White list for sites",
    )

    blacklist_sites = luigi.Parameter(
        default=law.NO_STR,
        description="Black list for sites",
    )

    limit = luigi.IntParameter(
        default=None,
        description="Limit numbers of file to create json",
    )

    redirector = luigi.ChoiceParameter(
        default="infn",
        choices=["infn", "fnal", "cern"],
        description="xrootd redirector in case sites are not found",
    )

    ncpus = luigi.Parameter(
        default="4",
        description="Number of CPUs to use for validation",
    )

    skipvalidation = luigi.BoolParameter(
        default=False,
        description="If true, the readability of files will not be validated",
    )

    DAS_campaign = luigi.Parameter(
        default=law.NO_STR,
        description="campaign info, specifying dataset name in DAS. If you are running with from_workflow option, please do campaign1,campaign2,campaign3 split by comma. E.g. Run2022C*Sep2023,Run2022D*Sep2023,Run3Summer22NanoAODv12-130X",
    )

    campaign = luigi.Parameter(
        description="campaign name (same as the campaign in runner.py)",
    )

    year = luigi.Parameter(
        default=None,
        description="year",
    )


@luigi.util.inherits(datasetparameters)
class FetchDatasets(BaseTask):
    """
    Create dataset json files.
    """

    def local_file_target(self, *path: str) -> law.LocalFileTarget:
        """Return a LocalFileTarget for the given path(s).
        Pass multiple path parts as separate arguments.

        :return: LocalFileTarget for the given path
        :rtype: law.LocalFileTarget
        """
        return law.LocalFileTarget(Path(os.environ["BTV_BASE"], *path))

    def output(self):
        """
        Return output targets based on the fetch.py logic:
        - If from_workflow: multiple JSON files for each sample
        - Otherwise: single JSON file with the specified output name
        """
        if self.from_workflow != law.NO_STR:
            from BTVNanoCommissioning.utils.sample import predefined_sample

            # Create a target for each sample in the workflow
            outputs = {}
            for sample in predefined_sample[self.from_workflow].keys():
                filename = f"metadata/{self.campaign}/{sample}_{self.campaign}_{self.year}_{self.from_workflow}.json"
                outputs[sample] = self.local_file_target(filename)
            return outputs
        else:
            # Single output file
            filename = f"metadata/{self.campaign}/{self.out}"
            return self.local_file_target(filename)

    def _create_args_namespace(self):
        """
        Create an argparse-like namespace object from Luigi parameters.
        This allows us to pass Luigi parameters to functions expecting argparse args.
        Converts law.NO_STR to None for compatibility with argparse behavior.
        """
        from types import SimpleNamespace

        # Helper function to convert law.NO_STR to None
        def convert_no_str(value):
            return None if value == law.NO_STR else value

        # Create a namespace with all parameters mapped to match argparse names
        args = SimpleNamespace(
            input=convert_no_str(self.inp),
            output=self.out,
            xrd=convert_no_str(self.xrd),
            from_path=self.from_path,
            from_dataset=self.from_dataset,
            from_workflow=convert_no_str(self.from_workflow),
            testfile=self.testfile,
            whitelist_sites=convert_no_str(self.whitelist_sites),
            blacklist_sites=convert_no_str(self.blacklist_sites),
            limit=self.limit,
            redirector=self.redirector,
            ncpus=self.ncpus,
            skipvalidation=self.skipvalidation,
            overwrite=False,  # output handling done by law and --remove-output
            DAS_campaign=convert_no_str(self.DAS_campaign),
            campaign=self.campaign,
            year=self.year,
        )

        return args

    def run(self):
        import importlib.util
        import sys
        from pathlib import Path

        # Get the path to the fetch.py script
        # This file is at: src/BTVNanoCommissioning/tasks/datasets.py
        # fetch.py is at: scripts/fetch.py
        # So we need to go up 3 levels from this file
        current_file = Path(__file__)
        fetch_script = current_file.parent.parent.parent.parent / "scripts" / "fetch.py"

        # Load the fetch module dynamically
        spec = importlib.util.spec_from_file_location("fetch", fetch_script)
        fetch_module = importlib.util.module_from_spec(spec)
        sys.modules["fetch"] = fetch_module
        spec.loader.exec_module(fetch_module)

        # Create args namespace that mimics argparse.Namespace
        args = self._create_args_namespace()

        # Call the main function from fetch.py
        fetch_module.main(args)
