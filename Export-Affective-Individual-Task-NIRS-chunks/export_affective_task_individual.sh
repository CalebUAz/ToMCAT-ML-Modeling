#!/bin/bash
'''
This script filters out NIRS data as chunks for affective task individual and renames the files according to the REDCap file.
RedCap file: REDCap_ToMCAT.csv has subject IDs for each imac.

Execution: ./export_affective_task_individual.sh <source_directory> <destination_directory> <REDCap_file>
<source_directory> is the labeled experiment data directory with filered_NIRS.csv (/space/calebshibu/Neurips_new)
<destination_directory> is user defined export location /space/calebshibu/
<REDCap_file> is present in /space/calebshibu/REDCap_ToMCAT.csv

e.g: ./export_affective_task_individual.sh /space/calebshibu/Neurips_new /space/calebshibu/Neurips_new/Affective_Task_Individual /space/calebshibu/REDcap_ToMCAT.csv
'''

# This script expects three arguments: the source directory, the destination directory, and the REDCap_ToMCAT.csv file.

# Check for required number of arguments
if [ "$#" -ne 3 ]; then
    echo "You must enter exactly 3 command line arguments"
    exit 1
fi

SRC_DIR=$1
DEST_DIR=$2
REDCAP_FILE=$3

# Check if source directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Source directory does not exist"
    exit 1
fi

# Check if REDCap file exists
if [ ! -f "$REDCAP_FILE" ]; then
    echo "REDCap file does not exist"
    exit 1
fi

# Loop over all experiment folders
for exp_dir in "$SRC_DIR"/exp_*; do
    # Check if directory
    if [ -d "$exp_dir" ]; then
        # Get the experiment ID from the directory name
        experiment_id="${exp_dir##*/}"
        # Create destination directory
        mkdir -p "$DEST_DIR/$experiment_id"
        # Create a log file in each experiment folder
        log_file="$DEST_DIR/$experiment_id/player_info.log"
        if touch "$log_file" ; then
            echo "Successfully created log file: $log_file"
        else
            echo "Failed to create log file: $log_file"
            continue
        fi
        echo "experiment_id: $experiment_id" | tee -a "$log_file"
        # Get Team_ID from REDCap file
        team_id=$(csvgrep -c "Experiment_ID" -m "$experiment_id" "$REDCAP_FILE" | csvcut -c "Team_ID" | tail -n +2)
        echo "Team_ID:$team_id" | tee -a "$log_file"
        # Loop over imac folders
        for imac_dir in "$exp_dir"/{lion,tiger,leopard}; do
            # Check if directory
            if [ -d "$imac_dir" ]; then
                csv_file="$imac_dir/NIRS_filtered.csv"
                # Check if csv file exists
                if [ -f "$csv_file" ]; then
                    # Get relative path for directory to maintain the same structure in destination
                    relative_dir="${imac_dir#$SRC_DIR}"
                    # Get the imac name from the path
                    imac_name="${relative_dir##*/}"
                    # Capitalize the first letter of imac_name
                    imac_name_capitalized="$(tr '[:lower:]' '[:upper:]' <<< ${imac_name:0:1})${imac_name:1}"
                    # Create destination directory for imac folders
                    mkdir -p "$DEST_DIR/${relative_dir%/*}"
                    echo "Filtering CSV file for $imac_name"
                    # Filter csv file
                    csvgrep -t -c "event_type" -m "affective_task_individual" "$csv_file" > "$DEST_DIR/${relative_dir%/*}/$imac_name.csv"
                    # Get new filename from REDCap file
                    new_filename=$(csvgrep -c "Experiment_ID" -m "$experiment_id" "$REDCAP_FILE" | csvcut -c "${imac_name_capitalized}_Subject_ID" | tail -n +2)
                    # If new filename is not empty, rename file
                    if [ -n "$new_filename" ]; then
                        mv "$DEST_DIR/${relative_dir%/*}/$imac_name.csv" "$DEST_DIR/${relative_dir%/*}/$new_filename.csv"
                        echo "$imac_name.csv:$new_filename.csv" | tee -a "$log_file"
                    fi
                fi
            fi
        done
    fi
done





