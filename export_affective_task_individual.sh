#!/bin/bash

# This script expects two arguments: the source and the destination folders.

# Check for required number of arguments
if [ "$#" -ne 2 ]; then
    echo "You must enter exactly 2 command line arguments"
    exit 1
fi

SRC_DIR=$1
DEST_DIR=$2

# Loop over all experiment folders
for exp_dir in "$SRC_DIR"/exp_*; do
    # Check if directory
    if [ -d "$exp_dir" ]; then
        # Loop over animal folders
        for animal_dir in "$exp_dir"/{lion,tiger,leopard}; do
            # Check if directory
            if [ -d "$animal_dir" ]; then
                csv_file="$animal_dir/NIRS_filtered.csv"
                # Check if csv file exists
                if [ -f "$csv_file" ]; then
                    # Get relative path for directory to maintain the same structure in destination
                    relative_dir="${animal_dir#$SRC_DIR}"
                    # Create destination directory
                    mkdir -p "$DEST_DIR/$relative_dir"
                    # Filter csv file and export it to the destination directory
                    csvgrep -t -c "event_type" -m "affective_task_individual" "$csv_file" > "$DEST_DIR/$relative_dir/exported_data.csv"
                fi
            fi
        done
    fi
done
