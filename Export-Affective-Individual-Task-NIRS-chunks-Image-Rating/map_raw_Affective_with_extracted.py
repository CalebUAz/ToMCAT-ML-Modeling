import os
import re
import argparse
import pandas as pd

"""
This script matches the raw affective csv files (/tomcat/data/raw/LangLab/experiments/study_3_pilot/group) with the extracted affective csv files (/space/calebshibu/Neurips_new/Affective_Task_Individual).
The raw affective csv files are located in the raw experiment folder.
The extracted affective csv files are located in the extracted experiment folder.
The matched files are saved in the export folder.
"""

def match_files(raw_folder, extracted_folder):
    raw_files = []
    extracted_files = []
    matched_files = {}

    # Get the list of CSV files in the raw experiment folder
    raw_affective_folder = os.path.join(raw_folder, "baseline_tasks", "affective")
    if os.path.isdir(raw_affective_folder):
        raw_files = [os.path.join(raw_affective_folder, f) for f in os.listdir(raw_affective_folder) if f.endswith(".csv")]

    # Get the list of CSV files in the extracted experiment folder
    for root, dirs, files in os.walk(extracted_folder):
        for file in files:
            if file.endswith(".csv"):
                extracted_files.append(os.path.join(root, file))

    # Match the files
    for raw_file in raw_files:
        match = re.search(r'individual_(\d+)_', raw_file)  # Find the file number using regular expressions
        if match:
            raw_file_number = match.group(1).lstrip("0")
            for extracted_file in extracted_files:
                extracted_file_number = os.path.splitext(os.path.basename(extracted_file))[0]
                if extracted_file_number.isdigit() and int(extracted_file_number) == int(raw_file_number):
                    matched_files[raw_file] = extracted_file
                    break

    return matched_files

def process_csv(raw_file_path, extracted_file_path, output_file_path):
    raw_affective = pd.read_csv(raw_file_path, delimiter=';')
    extracted_affective = pd.read_csv(extracted_file_path)

    # print("Raw Affective Columns:", raw_affective.columns)
    # print("Extracted Affective Columns:", extracted_affective.columns)
    
    # Perform the operations on the dataframes
    extracted_affective.drop("Unnamed: 0", axis=1, inplace=True)
    raw_affective.fillna(method='bfill', inplace=True)
    extracted_affective.fillna(method='bfill', inplace=True)

    # Convert 'human_readable_time' column in extracted_affective to datetime type
    extracted_affective['human_readable_time'] = pd.to_datetime(extracted_affective['human_readable_time'])

    # Perform the merge based on closest matching time
    merged_df = pd.merge_asof(extracted_affective.sort_values('unix_time'),
                              raw_affective.sort_values('time'),
                              left_on='unix_time',
                              right_on='time',
                              direction='nearest')
    # Save the merged_df to a csv file
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    merged_df.to_csv(output_file_path, index=False)
    
    return merged_df

def main():
    parser = argparse.ArgumentParser(description="Match files between raw and extracted experiment folders.")
    parser.add_argument("--raw-folder", required=True, help="Path to the raw experiment folder")
    parser.add_argument("--extracted-folder", required=True, help="Path to the extracted experiment folder")
    parser.add_argument("--export-folder", required=True, help="Path to the export folder")
    
    args = parser.parse_args()

    raw_folder_path = args.raw_folder
    extracted_folder_path = args.extracted_folder
    output_folder_path = args.export_folder

    # # Extract the 'exp_*' part from the extracted_folder_path
    # match = re.search(r'exp_.*', extracted_folder_path)
    # if match:
    #     exp_part = match.group(0)
    # else:
    #     raise ValueError("The extracted_folder_path doesn't contain an 'exp_*' part")

    # # Append the 'exp_*' part to the output_folder_path
    # output_folder_path = os.path.join(output_folder_path, exp_part)

    # Call the function to match the files
    matched_files = match_files(raw_folder_path, extracted_folder_path)

    # Process each matched pair of csv files
    for raw_file, extracted_file in matched_files.items():
        raw_file_path = os.path.join(raw_folder_path, 'baseline_tasks', 'affective', raw_file)
        extracted_file_path = os.path.join(extracted_folder_path, extracted_file)

       # Calculate the relative path of the extracted file path to the extracted folder path
        relative_extracted_path = os.path.relpath(extracted_file_path, start=extracted_folder_path)
        # Now the relative path can be joined with the output folder path
        output_file_path = os.path.join(output_folder_path, relative_extracted_path)

        merged_df = process_csv(raw_file_path, extracted_file_path, output_file_path)
        print(f"Processed and saved file: {output_file_path}")

if __name__ == "__main__":
    main()