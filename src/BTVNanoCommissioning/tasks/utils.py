import json
import subprocess

import numpy as np


def get_lumi(coffea_output, json_file, year) -> float:
    lumi, run = [], []
    for v in coffea_output.values():
        if isinstance(v, dict) and "lumi" in v and "run" in v:
            lumi.append(v["lumi"].value)
            run.append(v["run"].value)

    # Sort runs and keep lumisections matched
    run, lumi = np.concatenate(run), np.concatenate(lumi)
    sorted_indices = np.lexsort((lumi, run))  # Sort by run first, then lumi
    run = run[sorted_indices]
    lumi = lumi[sorted_indices]
    # Create dictionary with ls values for each run and convert to brilcalc format
    lumi_per_run = {
        str(r): lumi[run == r].astype(int).reshape(-1, 1).repeat(2, axis=1).tolist()
        for r in np.unique(run)
    }

    with open(json_file, "w") as outfile:
        json.dump(lumi_per_run, outfile, indent=4)

    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/LumiRecommendationsRun3
    if year in ["2022", "2023", "2024"]:
        brilcalc_cmd = f"source /cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env; brilcalc lumi --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -c web -i {json_file} -u /pb"
    elif year in ["2025"]:
        # Using recommended temporary Run 3 normtag
        brilcalc_cmd = f"source /cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env; brilcalc lumi --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json -c web -i {json_file} -u /pb --datatag online"
    elif year in ["2016", "2017", "2018"]:
        # Run 2 UL normtag
        brilcalc_cmd = f"source /cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env; brilcalc lumi --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -c web -i {json_file} -u /pb"
    else:
        print(f"WARNING: Unknown year '{year}', skipping brilcalc lumi calculation.")
        raise ValueError(f"Unknown year '{year}' for luminosity calculation.")
    brilcalc_output = subprocess.run(
        brilcalc_cmd,
        shell=True,
        capture_output=True,
        text=True,
        executable="/bin/bash",
    )
    if brilcalc_output.returncode != 0:
        raise RuntimeError(
            f"brilcalc command failed with error: {brilcalc_output.stderr}"
        )

    lumi = brilcalc_output.stdout
    lumi_in_pb = lumi[lumi.find("#Summary:") : lumi.find("#Check JSON:")]
    lumi_in_pb = float(lumi_in_pb.split("\n")[4].split("|")[-2])

    return lumi_in_pb
