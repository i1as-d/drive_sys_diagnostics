#!/bin/bash

# Determine the current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if duration argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <duration>"
    exit 1
fi

# Set duration
DURATION=$1

# Clear the data folder
rm -rf "$SCRIPT_DIR/../data"

# Create the data folder
mkdir -p "$SCRIPT_DIR/../data"

# File paths
ATOP_FILE="$SCRIPT_DIR/../data/atop_$(date +'%Y_%m_%d_%H_%M_%S')"
GPU_FILE="$SCRIPT_DIR/../data/nvidia_gpu_usage.txt"

echo "resources will be recorded in : "
echo $ATOP_FILE
echo "gpu readings will be recorded in : "
echo $GPU_FILE
echo ""

# Start atop recording, overwrite if exists, create if not exist
atop -w "$ATOP_FILE" -S -a 1 "$DURATION" &> /dev/null

# Start Nvidia GPU usage recording, overwrite if exists, create if not exist
timeout "$DURATION"s nvidia-smi dmon -s puc -d 1 -o DT -f "$GPU_FILE" -d "$DURATION" &> /dev/null

# Wait for the specified duration
sleep "$DURATION"

# Done
echo "Resource recording finished. Processing data ..."

# Parse atop to readable format and overwrite existing file
atopsar -r "$ATOP_FILE" -w > "$SCRIPT_DIR/../data/atop_report.csv"
