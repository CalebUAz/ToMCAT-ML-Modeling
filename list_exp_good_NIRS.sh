#!/bin/bash

#List all experiments with good NIRS data i.e. with bad channels

# Specify the path to the folder containing the experiment folders
experiment_folder="/path/to/experiment_folder"

# Loop through each experiment folder
for exp_folder in "$experiment_folder"/exp_*/; do
  # Check if the experiment folder contains the required subfolders
  if [ -d "${exp_folder}lion" ] && [ -d "${exp_folder}tiger" ] && [ -d "${exp_folder}leopard" ]; then
    # Check if the NIRS_channel_quality.csv file exists in each subfolder
    if [ -f "${exp_folder}lion/NIRS_channel_quality.csv" ] && [ -f "${exp_folder}tiger/NIRS_channel_quality.csv" ] && [ -f "${exp_folder}leopard/NIRS_channel_quality.csv" ]; then
      # Check if all rows in the status column have the value "good_channel"
      if awk -F',' '{ if ($3 != "good_channel") exit 1 }' "${exp_folder}lion/NIRS_channel_quality.csv" \
                  && awk -F',' '{ if ($3 != "good_channel") exit 1 }' "${exp_folder}tiger/NIRS_channel_quality.csv" \
                  && awk -F',' '{ if ($3 != "good_channel") exit 1 }' "${exp_folder}leopard/NIRS_channel_quality.csv"; then
        # Display the path of the experiment folder
        echo "$exp_folder"
      fi
    fi
  fi
done
