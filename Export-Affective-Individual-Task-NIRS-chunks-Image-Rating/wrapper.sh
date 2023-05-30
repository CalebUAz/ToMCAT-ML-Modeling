#!/bin/bash

'''
This script runs after /ToMCAT-ML-Modeling/Export-Affective-Individual-Task-NIRS-chunks. It merges scores from baseline task raw folder with results from Individual-Task-NIRS-chunks. 
Execution: ./wrapper.sh -r <raw_directory> -e <extracted_directory> -x <export_directory>

e.g: ./wrapper.sh -r /tomcat/data/raw/LangLab/experiments/study_3_pilot/group -e /space/calebshibu/Neurips_new/Affective_Task_Individual -x /space/calebshibu/Affective_Task_Individual_rating
'''
# Define an array of directories to ignore
ignore=("exp_2022_04_01_13" "exp_2022_04_22_09" "exp_2023_04_17_13" "exp_2023_04_18_14" "exp_2023_04_20_14" "exp_2023_04_21_10" "exp_2023_04_24_13" "exp_2023_04_27_14" "exp_2023_04_28_10" "exp_2023_05_01_13" "exp_2023_05_02_14" "exp_2023_05_03_10")

while getopts ":r:e:x:" opt; do
  case $opt in
    r) rawdir="$OPTARG"
    ;;
    e) extracteddir="$OPTARG"
    ;;
    x) exportdir="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Check if raw directory, extracted directory and export directory were provided
if [ -z "$rawdir" ] || [ -z "$extracteddir" ] || [ -z "$exportdir" ]; then
  echo "You must provide a raw directory with the -r flag, an extracted directory with the -e flag and an export directory with the -x flag"
  exit 1
fi

# Search for directories starting with exp_ under the raw directory
directories=($(ls -d $rawdir/exp_*))

# Define the function to execute the Python script
execute_python_script() {
    local dir="$1"
    local current_raw_folder="${rawdir}/${base_dir}"
    local current_extracted_folder="${extracteddir}/${base_dir}"
    local current_export_folder="${exportdir}/${base_dir}"

    # Create the directory if it does not exist
    mkdir -p "$current_export_folder"

    local error_log="$current_export_folder/error_log.txt"

    # Call python script and save stderr to error log
    if ! python3 map_raw_Affective_with_extracted.py --raw-folder "$current_raw_folder" --extracted-folder "$current_extracted_folder" --export-folder" "$current_export_folder" 2>> "$error_log"; then
        echo "Python script failed for $dir."
    fi
}

for dir in "${directories[@]}"; do
    # Extract the base directory name
    base_dir=$(basename "$dir")

    # Check if directory is in ignore list
    if [[ " ${ignore[@]} " =~ " ${base_dir} " ]]; then
        # If directory is in ignore list, skip to next iteration
        continue
    fi

    # Print the command before executing
    echo "Running: python3 map_raw_Affective_with_extracted.py echo "Running: python3 map_raw_Affective_with_extracted.py --raw-folder \"${rawdir}/${base_dir}\" --extracted-folder \"${extracteddir}/${base_dir}\" --export-folder \"${exportdir}/${base_dir}\""

    # Start a separate process to execute the Python script
    execute_python_script "$dir" &
done

# Wait for all child processes to finish
wait
