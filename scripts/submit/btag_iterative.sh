#!/bin/bash

# Default values
WORKFLOW="btag_iterative_sf"
CHANNEL=""
LIMIT_MC=""
LIMIT_DATA=""
MAX_MC=""
MAX_DATA=""
EXECUTOR_DEFAULT="parsl/condor"
EXECUTOR=""
SCALEOUT=200
ISSYST="False"
TEST_MODE=false
TEST_MAX=""  # Add this new variable
OVERWRITE=""
# CHUNKSIZE=250000
CHUNKSIZE=1000
VERSION="no_version"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -c CAMPAIGN     Campaign name
    -y YEAR         Year
    -r RUN          Run period
    -w WORKFLOW     Workflow name (default: $WORKFLOW)
    -p CHANNEL      Channel selector for workflow (default: )
    -v VERSION      Output Version
    -m LIMIT        Limit for MC processing
    -d LIMIT        Limit for data processing
    -e EXECUTOR     Executor type (default: $EXECUTOR)
    -n SCALEOUT     Scaleout value (default: $SCALEOUT)
    -i ISSYST       System flag (default: $ISSYST)
    -t NUMBER       Enable test mode with max files limit
    -l LUMI         Luminosity to scale plots
    -o              Enable overwrite mode
    -h              Show this help message

Examples:
    $0 -t 5
    $0 -c Summer23 -y 2023 -m 5
    $0 -e iterative -n 50 -o
EOF
}

# Parse command line options
while getopts ":c:y:r:w:v:m:d:e:n:i:t:p:l:oh" option; do
    case $option in
        c)
            CAMPAIGN="$OPTARG"
            ;;
        y)
            YEAR="$OPTARG"
            ;;
        r)
            RUN_PERIOD="$OPTARG"
            ;;
        w)
            WORKFLOW="$OPTARG"
            ;;
        p)  CHANNEL="$OPTARG"
            ;;
        v)
            VERSION="$OPTARG"
            ;;
        m)
            LIMIT_MC="$OPTARG"
            ;;
        d)
            LIMIT_DATA="$OPTARG"
            ;;
        e)
            EXECUTOR="$OPTARG"
            ;;
        n)
            SCALEOUT="$OPTARG"
            ;;
        i)
            ISSYST="$OPTARG"
            ;;
        t)
            TEST_MODE=true
            TEST_MAX="$OPTARG"
            # Validate that it's a number
            if ! [[ "$TEST_MAX" =~ ^[0-9]+$ ]]; then
                echo "Error: Test mode argument must be a positive integer." >&2
                usage
                exit 1
            fi
            ;;
        l)
            LUMI="$OPTARG"
            ;;
        o)
            OVERWRITE=true
            ;;
        h)
            usage
            exit 0
            ;;
        :)
            echo "Error: Option -$OPTARG requires an argument." >&2
            usage
            exit 1
            ;;
        \?)
            echo "Error: Invalid option -$OPTARG" >&2
            usage
            exit 1
            ;;
    esac
done

# Validate mandatory arguments
if [ -z "$CAMPAIGN" ]; then
    echo "Error: Campaign (-c) is required." >&2
    usage
    exit 1
fi

if [ -z "$YEAR" ]; then
    echo "Error: Year (-y) is required." >&2
    usage
    exit 1
fi

if [ -z "$RUN_PERIOD" ]; then
    echo "Error: Run period (-r) is required." >&2
    usage
    exit 1
fi

# Apply test mode settings if enabled

# Set executor logic
if [ "$TEST_MODE" = true ]; then
    echo "Running in test mode with max $TEST_MAX files"
    LIMIT_MC=1
    LIMIT_DATA=1
    MAX_MC="$TEST_MAX"
    MAX_DATA="$TEST_MAX"
    # SCALEOUT=32
    VERSION="${VERSION}_test"
    # If executor not set by -e, use iterative
    if [ -z "$EXECUTOR" ]; then
        EXECUTOR="iterative"
    fi
fi

# If not set by test mode or -e, use default
if [ -z "$EXECUTOR" ]; then
    EXECUTOR="$EXECUTOR_DEFAULT"
fi

if [ $RUN_PERIOD = "all" ]; then
    RUN_PERIOD=(
        "RunC" "RunD" "RunE" "RunF" "RunG" "RunH" "RunI"
    )
else
    RUN_PERIOD=("$RUN_PERIOD")
fi

# https://superuser.com/questions/461981/how-do-i-convert-a-bash-array-variable-to-a-string-delimited-with-newlines
runs=$( IFS=$','; echo "${RUN_PERIOD[*]}" )
output_data=""
for run in "${RUN_PERIOD[@]}"; do
    output_data+="hists_data_${run}/hists_data_${run}.coffea,"
done
output_data=${output_data%,}  # remove trailing comma


echo "Configuration:"
echo "  Campaign: $CAMPAIGN"
echo "  Year: $YEAR"
echo "  Run: $runs"
echo "  Workflow: $WORKFLOW"
echo "  Channel: $CHANNEL"
echo "  Version: $VERSION"
echo "  Limit MC: $LIMIT_MC"
echo "  Limit Data: $LIMIT_DATA"
echo "  Max MC: $MAX_MC"
echo "  Max Data: $MAX_DATA"
echo "  Executor: $EXECUTOR"
echo "  Scaleout: $SCALEOUT"
echo "  IsSyst: $ISSYST"
echo "  Overwrite: $OVERWRITE"

WORKFLOW_CHANNEL="${WORKFLOW}"
if [[ "${CHANNEL}" != "" ]]; then
    WORKFLOW_CHANNEL="${WORKFLOW}_${CHANNEL}"
fi

# OUTPUTDIR="/net/data_cms3a-1/fzinn/BTV/btag_sf/nobackup/${CAMPAIGN}/${WORKFLOW_CHANNEL}/${VERSION}"
OUTPUTDIR="/net/data_cms3a-1/BTV/btag_sf/${CAMPAIGN}/${WORKFLOW_CHANNEL}/${VERSION}"

# execution
for json in dy ttbar WZ singletop; do
    python runner.py \
        --json "metadata/${CAMPAIGN}/btag_iterative_sf/${json}.json" \
        --workflow "${WORKFLOW_CHANNEL}" \
        --campaign "$CAMPAIGN" \
        --year "$YEAR" \
        --chunk "$CHUNKSIZE" \
        --outputdir "$OUTPUTDIR" \
        --output "hists_MC_${json}.coffea" \
        --executor "$EXECUTOR" \
        --scaleout "$SCALEOUT" \
        --skipbadfiles \
        ${LIMIT_MC:+--limit "$LIMIT_MC"} \
        ${MAX_MC:+--max "$MAX_MC"} \
        --isSyst "$ISSYST" \
        ${OVERWRITE:+--overwrite} \
        --splitjobs
    echo
done

for run in "${RUN_PERIOD[@]}"; do
    python runner.py \
        --json "metadata/${CAMPAIGN}/btag_iterative_sf/data_${run}.json" \
        --workflow "${WORKFLOW_CHANNEL}" \
        --campaign "$CAMPAIGN" \
        --year "$YEAR" \
        --chunk "$CHUNKSIZE" \
        --outputdir "$OUTPUTDIR" \
        --output "hists_data_${run}.coffea" \
        --executor "$EXECUTOR" \
        --scaleout "$SCALEOUT" \
        --skipbadfiles \
        ${LIMIT_DATA:+--limit "$LIMIT_DATA"} \
        ${MAX_DATA:+--max "$MAX_DATA"} \
        --isSyst "$ISSYST" \
        ${OVERWRITE:+--overwrite} \
        --splitjobs
    echo
done

if [ "$TEST_MODE" = false ]; then
    basedir=$(pwd)
    cd "$OUTPUTDIR"
    python ${basedir}/scripts/plotdataMC.py \
        -p btag_iterative_sf_mumu \
        -v all \
        -i ${output_data},hists_MC_dy/hists_MC_dy.coffea,hists_MC_ttbar/hists_MC_ttbar.coffea,hists_MC_WZ/hists_MC_WZ.coffea,hists_MC_singletop/hists_MC_singletop.coffea \
        --lumi $LUMI \
        --log \
        --split sample


    python ${basedir}/scripts/plotdataMC.py \
        -p btag_iterative_sf_mumu \
        -v all \
        -i hists_data/hists_data.coffea,hists_MC_dy/hists_MC_dy.coffea,hists_MC_ttbar/hists_MC_ttbar.coffea,hists_MC_WZ/hists_MC_WZ.coffea,hists_MC_singletop/hists_MC_singletop.coffea \
        --lumi $LUMI \
        --split sample

    python ${basedir}/scripts/plot_histograms_iterative_btagSF.py --lumi $LUMI --log-level info
fi