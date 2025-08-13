#!/bin/bash
# filepath: /home/home1/institut_3a/zinn/analyses/BTV/BTVNanoCommissioning/scripts/submit/btag_iterative.sh

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
CHUNKSIZE=250000
VERSION="no_version"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -c CAMPAIGN      Campaign name
    -y YEAR         Year
    -w WORKFLOW     Workflow name (default: $WORKFLOW)
    -p CHANNEL      Channel selector for workflow (default: )
    -v VERSION      Output Version
    -m LIMIT        Limit for MC processing
    -d LIMIT        Limit for data processing
    -e EXECUTOR     Executor type (default: $EXECUTOR)
    -n SCALEOUT     Scaleout value (default: $SCALEOUT)
    -i ISSYST       System flag (default: $ISSYST)
    -t NUMBER       Enable test mode with max files limit
    -o              Enable overwrite mode
    -h              Show this help message

Examples:
    $0 -t 5
    $0 -c Summer23 -y 2023 -m 5
    $0 -e iterative -n 50 -o
EOF
}

# Parse command line options
while getopts ":c:y:w:v:m:d:e:n:i:t:p:oh" option; do
    case $option in
        c)
            CAMPAIGN="$OPTARG"
            ;;
        y)
            YEAR="$OPTARG"
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

# Apply test mode settings if enabled

# Set executor logic
if [ "$TEST_MODE" = true ]; then
    echo "Running in test mode with max $TEST_MAX files"
    LIMIT_MC=1
    LIMIT_DATA=1
    MAX_MC="$TEST_MAX"
    MAX_DATA="$TEST_MAX"
    SCALEOUT=32
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

echo "Configuration:"
echo "  Campaign: $CAMPAIGN"
echo "  Year: $YEAR"
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

OUTPUTDIR="/net/data_cms3a-1/fzinn/BTV/btag_sf/nobackup/${CAMPAIGN}/${WORKFLOW_CHANNEL}/${VERSION}"

# execution
python runner.py \
    --json "metadata/${CAMPAIGN}/data_${CAMPAIGN}_${YEAR}_${WORKFLOW}.json" \
    --workflow "${WORKFLOW_CHANNEL}" \
    --campaign "$CAMPAIGN" \
    --year "$YEAR" \
    --chunk "$CHUNKSIZE" \
    --outputdir "$OUTPUTDIR" \
    --output "hists_data.coffea" \
    --executor "$EXECUTOR" \
    --scaleout "$SCALEOUT" \
    --skipbadfiles \
    ${LIMIT_DATA:+--limit "$LIMIT_DATA"} \
    ${MAX_DATA:+--max "$MAX_DATA"} \
    --isSyst "$ISSYST" \
    ${OVERWRITE:+--overwrite}

python runner.py \
    --json "metadata/${CAMPAIGN}/MC_${CAMPAIGN}_${YEAR}_${WORKFLOW}.json" \
    --workflow "${WORKFLOW_CHANNEL}" \
    --campaign "$CAMPAIGN" \
    --year "$YEAR" \
    --chunk "$CHUNKSIZE" \
    --outputdir "$OUTPUTDIR" \
    --output "hists_MC.coffea" \
    --executor "$EXECUTOR" \
    --scaleout "$SCALEOUT" \
    --skipbadfiles \
    ${LIMIT_MC:+--limit "$LIMIT_MC"} \
    ${MAX_MC:+--max "$MAX_MC"} \
    --isSyst "$ISSYST" \
    ${OVERWRITE:+--overwrite}
