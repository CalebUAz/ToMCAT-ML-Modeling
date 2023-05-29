#!/bin/bash

#List all experiments with good NIRS data i.e. with bad channels

# Usage: ./script.sh -p /path/to/directory

# Parse command line arguments
while getopts ":p:" opt; do
  case ${opt} in
    p ) 
      # Assign the provided path to a variable
      parent_directory=$OPTARG
      ;;
    \? ) 
      echo "Usage: cmd -p path"
      ;;
  esac
done

# If no directory is provided, display a usage message
if [ -z "${parent_directory}" ]; then
    echo "You must provide a directory path with -p."
    exit 1
fi

# Change directory to the parent_directory
cd "${parent_directory}"

# Iterate over each experiment directory
for exp_folder in exp_*; do
  all_good=true
  # Iterate over each animal folder
  for animal in lion tiger leopard; do
    csv_file="${exp_folder}/${animal}/NIRS_channel_quality.csv"
    # Check if all status are good_channel using awk
    if ! awk -F',' '{if (NR>1 && $3 != "good_channel") exit 1}' "${csv_file}"; then
      all_good=false
      break
    fi
  done
  # If all statuses in all csv files are good_channel, print the exp_folder path
  if ${all_good}; then
    echo "${parent_directory}/${exp_folder}"
  fi
done