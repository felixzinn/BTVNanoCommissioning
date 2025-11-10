import json
import os

import law
import luigi
from coffea.util import save

from BTVNanoCommissioning.tasks.base import BaseTask
from BTVNanoCommissioning.tasks.datasets import FetchDatasets
from BTVNanoCommissioning.tasks.processing_utils import (
    create_parsl_condor_config,
    create_processor_instance,
    run_coffea_processor,
)
from BTVNanoCommissioning.workflows import workflows


class processingparameters(luigi.Config):
    """
    Configuration class for processing (runner.py) parameters.
    """

    json = luigi.Parameter(
        description="JSON file containing dataset and file locations",
    )

    workflow = luigi.ChoiceParameter(
        choices=sorted(workflows.keys()),
        description="Which processor to run",
    )

    campaign = luigi.Parameter(
        description="Dataset campaign, change the corresponding correction files",
    )

    year = luigi.Parameter(
        description="Year",
    )

    chunksize = luigi.IntParameter(
        default=400000,
        description="Number of events per process chunk",
    )

    outputdir = luigi.Parameter(
        default=law.NO_STR,
        description="Output directory name",
    )

    output_file = luigi.Parameter(
        default="hists.coffea",
        description="Output histogram filename",
    )

    executor = luigi.ChoiceParameter(
        default="futures",
        choices=[
            "iterative",
            "futures",
            "parsl/slurm",
            "parsl/condor",
            "parsl/condor/naf_lite",
            "dask/condor",
            "dask/condor/brux",
            "dask/slurm",
            "dask/lpc",
            "dask/lxplus",
            "dask/casa",
            "condor_standalone",
        ],
        description="The type of executor to use",
    )

    scaleout = luigi.IntParameter(
        default=200,
        description="Number of nodes to scale out to if using slurm/condor",
    )

    coffea_workers = luigi.IntParameter(
        default=3,
        description="Number of workers (cores/threads) to use for multi-worker executors",
    )

    memory = luigi.FloatParameter(
        default=4.0,
        description="Memory per worker in GB",
    )

    skipbadfiles = luigi.BoolParameter(
        default=False,
        description="Skip bad files",
    )

    limit = luigi.IntParameter(
        default=None,
        description="Limit to the first N files of each dataset in sample JSON",
    )

    max = luigi.IntParameter(
        default=None,
        description="Max number of chunks to run in total",
    )

    isSyst = luigi.ChoiceParameter(
        default="False",
        choices=[
            "False",
            "all",
            "weight_only",
            "JERC_full",
            "JERC_reduced",
            "JERC_total",
            "JP_MC",
        ],
        description="Run with systematics (default: False), weights_only (no JERC uncertainties included)",
    )

    splitjobs = luigi.BoolParameter(
        default=False,
        description="Split processing and merging",
    )


@luigi.util.inherits(processingparameters)
class ProcessDatasets(BaseTask):
    """
    Process datasets using runner.py (coffea processor).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.test:
            self.limit = 1
            self.max = 1

    def requires(self):
        return FetchDatasets.req(self, out=self.json)

    def store_parts(self):
        return super().store_parts() + (self.campaign, self.workflow)

    def output(self):
        # Determine output path based on outputdir and output parameters
        if self.outputdir != law.NO_STR:
            outdir = self.outputdir
            # if self.is_
            return law.LocalFileTarget(f"{outdir}/{self.output_file}")
        else:
            return self.local_file_target(self.output_file)

    def setup_x509_proxy(self):
        """
        Setup x509 proxy for xrootd access.

        :return: Path to the x509 proxy file
        :rtype: str
        """
        try:
            _x509_localpath = (
                [
                    line
                    for line in os.popen("voms-proxy-info").read().split("\n")
                    if line.startswith("path")
                ][0]
                .split(":")[-1]
                .strip()
            )
        except Exception:
            raise RuntimeError(
                "x509 proxy could not be parsed, try creating it with 'voms-proxy-init'"
            )
        _x509_path = os.environ["HOME"] + f"/.{_x509_localpath.split('/')[-1]}"
        os.system(f"cp {_x509_localpath} {_x509_path}")
        return _x509_path

    def setup_job_script_prologue(self, x509_path):
        """
        Setup job script prologue and condor extra commands for parsl executor.

        :param x509_path: Path to the x509 proxy file
        :type x509_path: str
        :return: Tuple of (job_script_prologue, condor_extra)
        :rtype: tuple[list[str], list[str]]
        """
        # Setup job script prologue
        job_script_prologue = [
            "export XRD_RUNFORKHANDLER=1",
            f"export X509_USER_PROXY={x509_path}",
            f"export X509_CERT_DIR={os.environ['X509_CERT_DIR']}",
            f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
        ]

        pathvar = [i for i in os.environ["PATH"].split(":") if "envs/btv_coffea" in i][
            0
        ]
        condor_extra = [
            f"source {os.environ['HOME']}/.bashrc",
            f"cd {os.getcwd()}",
        ]
        condor_extra.insert(0, f"export PATH={pathvar}:$PATH")

        # Check for conda/micromamba
        conda_check = os.system("command -v conda") == 0
        mamba_check = os.system("command -v micromamba") == 0

        if conda_check and mamba_check:
            use_conda = True
            if use_conda:
                condor_extra.append(f"conda activate {os.environ['CONDA_PREFIX']}")
            else:
                condor_extra.append(f"micromamba activate {os.environ['MAMBA_EXE']}")
        elif conda_check:
            condor_extra.append(f"conda activate {os.environ['CONDA_PREFIX']}")
        elif mamba_check:
            condor_extra.append(f"micromamba activate {os.environ['MAMBA_EXE']}")

        return job_script_prologue, condor_extra

    def load_samples(self):
        """
        Load samples from the JSON file specified in the configuration.

        :return: Dictionary of samples
        :rtype: dict
        """

        samples = self.input().load()
        ret = {}
        if self.limit is not None:
            for key, sample in samples.items():
                ret[key] = sample[: self.limit]
        return ret

    def run(self):
        # Load samples from JSON
        samples = self.load_samples()

        # Create processor instance
        processor_instance = create_processor_instance(
            workflow=self.workflow,
            campaign=self.campaign,
            year=self.year,
            isSyst=self.isSyst,
            chunksize=self.chunksize,
        )

        # Setup execution based on executor type
        if self.executor in ["futures", "iterative"]:
            # Use futures or iterative executor
            output = run_coffea_processor(
                samples=samples,
                processor_instance=processor_instance,
                executor_type=self.executor,
                skipbadfiles=self.skipbadfiles,
                workers=self.coffea_workers,
                chunksize=self.chunksize,
                maxchunks=self.max,
            )

        elif "parsl" in self.executor:
            # Use parsl executor
            import parsl

            # Setup x509 proxy and job script prologue
            x509_path = self.setup_x509_proxy()
            job_script_prologue, condor_extra = self.setup_job_script_prologue(
                x509_path
            )

            # Create and load parsl configuration
            htex_config = create_parsl_condor_config(
                workers=self.coffea_workers,
                memory=self.memory,
                scaleout=self.scaleout,
                job_script_prologue=job_script_prologue,
                condor_extra=condor_extra,
                splitjobs=self.splitjobs,
            )
            parsl.load(htex_config)

            # Run the job
            output = run_coffea_processor(
                samples=samples,
                processor_instance=processor_instance,
                executor_type="parsl",
                skipbadfiles=self.skipbadfiles,
                chunksize=self.chunksize,
                maxchunks=self.max,
                splitjobs=self.splitjobs,
            )

        else:
            raise NotImplementedError(f"Executor {self.executor} not supported")

        # Save output
        out = self.output()
        out.makedirs()
        save(output, out.path)
