"""
Utility functions for running coffea processors with different executors.

This module provides standalone functions for configuring and running
coffea processors that can be reused across different tasks and scripts.
"""

from coffea import processor
from coffea.nanoevents import PFNanoAODSchema


def parsl_retry_handler(exception, task_record):
    """
    Retry handler for parsl executor to handle ManagerLost exceptions.

    :param exception: The exception that was raised
    :param task_record: The task record from parsl
    :return: Retry delay in seconds
    :rtype: float
    """
    from parsl.executors.high_throughput.interchange import ManagerLost

    if isinstance(exception, ManagerLost):
        return 0.1
    else:
        return 1


def create_parsl_condor_config(
    workers,
    memory,
    scaleout,
    job_script_prologue,
    condor_extra,
    splitjobs=False,
    retries=25,
    walltime="03:00:00",
):
    """
    Create a parsl configuration for HTCondor execution.

    :param workers: Number of cores/threads per slot
    :type workers: int
    :param memory: Memory per slot in GB
    :type memory: float
    :param scaleout: Number of nodes to scale out to
    :type scaleout: int
    :param job_script_prologue: List of commands to run before job execution
    :type job_script_prologue: list[str]
    :param condor_extra: List of extra condor setup commands
    :type condor_extra: list[str]
    :param splitjobs: Whether to split into separate run and merge executors
    :type splitjobs: bool
    :param retries: Number of retries for failed tasks
    :type retries: int
    :param walltime: Maximum walltime for jobs
    :type walltime: str
    :return: Parsl configuration object
    :rtype: parsl.config.Config
    """
    from parsl.addresses import address_by_query
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import CondorProvider

    worker_init = "\n".join(job_script_prologue + condor_extra)

    if splitjobs:
        htex_config = Config(
            executors=[
                HighThroughputExecutor(
                    label="run",
                    address=address_by_query(),
                    max_workers=1,
                    provider=CondorProvider(
                        nodes_per_block=1,
                        cores_per_slot=workers,
                        mem_per_slot=memory,
                        init_blocks=scaleout,
                        max_blocks=scaleout + 10,
                        worker_init=worker_init,
                        walltime=walltime,
                    ),
                ),
                HighThroughputExecutor(
                    label="merge",
                    address=address_by_query(),
                    max_workers=1,
                    provider=CondorProvider(
                        nodes_per_block=1,
                        cores_per_slot=workers,
                        mem_per_slot=memory,
                        init_blocks=scaleout,
                        max_blocks=scaleout + 10,
                        worker_init=worker_init,
                        walltime=walltime,
                    ),
                ),
            ],
            retries=retries,
            retry_handler=parsl_retry_handler,
        )
    else:
        htex_config = Config(
            executors=[
                HighThroughputExecutor(
                    label="coffea_parsl_condor",
                    address=address_by_query(),
                    max_workers=1,
                    provider=CondorProvider(
                        nodes_per_block=1,
                        cores_per_slot=workers,
                        mem_per_slot=memory,
                        init_blocks=scaleout,
                        max_blocks=scaleout + 10,
                        worker_init=worker_init,
                        walltime=walltime,
                    ),
                )
            ],
            retries=retries,
        )

    return htex_config


def run_coffea_processor(
    samples,
    processor_instance,
    executor_type="futures",
    skipbadfiles=False,
    workers=3,
    chunksize=400000,
    maxchunks=None,
    xrootdtimeout=900,
    splitjobs=False,
):
    """
    Run coffea processor with specified executor type.

    Supports futures, iterative, and parsl executors. For parsl executor,
    the configuration must already be loaded before calling this function.

    :param samples: Dictionary of samples with file paths
    :type samples: dict
    :param processor_instance: Coffea processor instance
    :type processor_instance: coffea.processor.ProcessorABC
    :param executor_type: Type of executor ("futures", "iterative", or "parsl")
    :type executor_type: str
    :param skipbadfiles: Whether to skip bad files
    :type skipbadfiles: bool
    :param workers: Number of workers/threads (used for futures/iterative only)
    :type workers: int
    :param chunksize: Number of events per chunk
    :type chunksize: int
    :param maxchunks: Maximum number of chunks to process
    :type maxchunks: int or None
    :param xrootdtimeout: Timeout for xrootd connections in seconds (futures/iterative only)
    :type xrootdtimeout: int
    :param splitjobs: Whether jobs are split into run and merge executors (parsl only)
    :type splitjobs: bool
    :return: Processor output
    """
    # Build common executor arguments
    executor_args = {
        "skipbadfiles": skipbadfiles,
        "schema": PFNanoAODSchema,
    }

    # Select executor and add specific arguments
    if executor_type == "iterative":
        _exec = processor.iterative_executor
        executor_args.update(
            {
                "workers": workers,
                "xrootdtimeout": xrootdtimeout,
            }
        )
    elif executor_type == "futures":
        _exec = processor.futures_executor
        executor_args.update(
            {
                "workers": workers,
                "xrootdtimeout": xrootdtimeout,
            }
        )
    elif executor_type == "parsl":
        _exec = processor.parsl_executor
        if splitjobs:
            executor_args.update(
                {
                    "merging": True,
                    "merges_executors": ["merge"],
                    "jobs_executors": ["run"],
                    "config": None,
                }
            )
        else:
            executor_args.update(
                {
                    "config": None,
                }
            )
    else:
        raise ValueError(
            f"Unknown executor type: {executor_type}. "
            "Supported types are: 'futures', 'iterative', 'parsl'"
        )

    # Run the processor
    output = processor.run_uproot_job(
        samples,
        treename="Events",
        processor_instance=processor_instance,
        executor=_exec,
        executor_args=executor_args,
        chunksize=chunksize,
        maxchunks=maxchunks,
    )

    return output


def create_processor_instance(
    workflow,
    campaign,
    year,
    isSyst,
    chunksize,
    isArray=False,
    noHist=False,
):
    """
    Create a workflow processor instance.

    :param workflow: Workflow name
    :type workflow: str
    :param campaign: Dataset campaign
    :type campaign: str
    :param year: Year
    :type year: str
    :param isSyst: Systematics option ("False", "all", "weight_only", etc.)
    :type isSyst: str or bool
    :param chunksize: Number of events per chunk
    :type chunksize: int
    :param isArray: Whether to output arrays
    :type isArray: bool
    :param noHist: Whether to skip histogram output
    :type noHist: bool
    :return: Processor instance
    """
    from BTVNanoCommissioning.workflows import workflows

    # Convert isSyst string to bool if needed
    if isinstance(isSyst, str) and isSyst == "False":
        isSyst = False

    processor_instance = workflows[workflow](
        year=year,
        campaign=campaign,
        isSyst=isSyst,
        isArray=isArray,
        noHist=noHist,
        chunksize=chunksize,
    )

    return processor_instance
