#!/bin/bash

#List all experiments with good NIRS data i.e. with bad channels

# Usage: ./script.sh -p /path/to/directory

while getopts ":p:" opt; do
  case ${opt} in
    p ) 
      parent_directory=$OPTARG
      ;;
    \? ) 
      echo "Usage: cmd -p path"
      ;;
  esac
done

if [ -z "${parent_directory}" ]; then
    echo "You must provide a directory path with -p."
    exit 1
fi

cd "${parent_directory}"

for exp_folder in exp_*; do
  all_good=true
  for animal in lion tiger leopard; do
    csv_file="${exp_folder}/${animal}/NIRS_channel_quality.csv"
    # Check if the file exists before processing it with awk
    if [ -f "${csv_file}" ]; then
      if ! awk -F',' '{if (NR>1 && $3 != "good_channel") exit 1}' "${csv_file}"; then
        all_good=false
        break
      fi
    else
      echo "File not found: ${csv_file}"
      all_good=false
      break
    fi
  done
  if ${all_good}; then
    echo "${parent_directory}/${exp_folder}"
  fi
done